import os
import json
import aiohttp
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import psycopg_pool
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
LLM_MODEL = os.getenv('LLM_MODEL')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS'))
SITEMAP_URL = os.getenv('SITEMAP_URL')
SOURCE_NAME = os.getenv('SOURCE_NAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

if not SITEMAP_URL or not LLM_MODEL or not EMBEDDING_MODEL or not EMBEDDING_DIMENSIONS or not DB_HOST or not DB_PORT or not DB_USER or not DB_PASSWORD or not DB_NAME:
    raise ValueError('Set environment variables')

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using local LLM."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                    ],
                    "stream": False,
                    "format": "json"
                }
            ) as response:
                if response.status == 200:
                    # Print response headers for debugging
                    print("Response headers:", response.headers)
                    
                    # Get raw response text first
                    raw_text = await response.text()
                    
                    try:
                        result = json.loads(raw_text)
                        content = result.get("message", {}).get("content", "{}")
                        
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}")
                        return {"title": "Error parsing title", "summary": "Error parsing summary"}
                else:
                    print(f"Error calling Ollama: {response.status}")
                    print("Error response:", await response.text())
                    return {"title": "Error processing title", "summary": "Error processing summary"}
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}  # Use appropriate embedding model
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["embedding"]
                else:
                    print(f"Error getting embedding: {response.status}")
                    return [0] * EMBEDDING_DIMENSIONS  # Adjust dimension based on model (likely smaller than OpenAI)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * EMBEDDING_DIMENSIONS  # Adjust dimension based on your local model # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str, pool) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": SOURCE_NAME,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk, pool):
    """Insert a processed chunk into Postgres."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Ensure vector dimensions match
                embedding = chunk.embedding
                if len(embedding) != EMBEDDING_DIMENSIONS:
                    print(f"Embedding dimension mismatch for chunk {chunk.chunk_number} of {chunk.url}")
                    # Adjust to exactly EMBEDDING_DIMENSIONS dimensions
                    if len(embedding) < EMBEDDING_DIMENSIONS:
                        print(f"Embedding is too short, padding with zeros")
                        embedding = embedding + [0] * (EMBEDDING_DIMENSIONS - len(embedding))
                    else:
                        print(f"Embedding is too long, truncating to {EMBEDDING_DIMENSIONS} dimensions")
                        embedding = embedding[:EMBEDDING_DIMENSIONS]
                
                await cur.execute(
                    """
                    INSERT INTO site_pages 
                    (url, chunk_number, title, summary, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk.url,
                        chunk.chunk_number,
                        chunk.title,
                        chunk.summary,
                        chunk.content,
                        json.dumps(chunk.metadata),
                        embedding
                    )
                )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return True
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None
    


async def process_and_store_document(url: str, markdown: str, pool):
    """Process a document and store its chunks in parallel."""
    print(f"Started processing {url}")
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, pool) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk, pool) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], pool, max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown.raw_markdown, pool)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_docs_urls() -> List[str]:
    """Get URLs from given docs sitemap."""
    try:
        response = requests.get(SITEMAP_URL)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []
    
async def clear_existing_data(pool):
    """Clear all existing data from the site_pages table."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("TRUNCATE TABLE site_pages")
        print("Cleared existing data")
    except Exception as e:
        print(f"Error clearing data: {e}")

async def main():
    # Initialize DB connection pool
    pool = await init_db_connection()
    
    # Clear existing data
    await clear_existing_data(pool)
    
    # Get URLs from given docs sitemap
    urls = get_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls, pool)

async def init_db_connection():
    # Initialize database connection pool
    return psycopg_pool.AsyncConnectionPool(
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}",
        min_size=1,
        max_size=10
    )
    
if __name__ == "__main__":
    asyncio.run(main())
from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import aiohttp
import os
import json

from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
import asyncpg
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from typing_extensions import AsyncGenerator
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')

client = OpenAIModel(
    model_name=LLM_MODEL, provider=OpenAIProvider(base_url=f'{OLLAMA_BASE_URL}/v1')
)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    pool: asyncpg.Pool
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

model = OpenAIModel(LLM_MODEL, provider=client)

# Create agent with model specified by name
agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from Ollama."""
    logfire.info("Starting get_embedding", text_length=len(text))
    try:
        # Use direct Ollama API for embeddings
        logfire.info("Using direct Ollama API for embeddings")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result["embedding"]
                    logfire.info("Successfully retrieved embedding via Ollama API", embedding_length=len(embedding))
                    return embedding
                else:
                    logfire.error("Error with Ollama API", status=response.status)
                    logfire.warning("Returning zero vector")
                    return [0] * 768  # Return zero vector with appropriate dimension
    except Exception as e:
        logfire.error("Error with Ollama embedding API", error=str(e))
        logfire.warning("Returning zero vector")
        return [0] * 768  # Return zero vector on error

@agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the database pool and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    logfire.info("Starting retrieve_relevant_documentation", user_query=user_query)
    try:
        # Get the embedding for the query
        logfire.info("Getting embedding for query")
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Convert embedding to JSON string for pgvector query
        embedding_json = json.dumps(query_embedding)
        logfire.info("Converted embedding to JSON for pgvector query")
        
        # Query the database directly using pgvector
        logfire.info("Executing pgvector similarity search")
        rows = await ctx.deps.pool.fetch(
            """
            SELECT id, url, title, summary, content, metadata 
            FROM site_pages 
            WHERE metadata->>'source' = 'pydantic_ai_docs'
            ORDER BY embedding <=> $1
            LIMIT 5
            """,
            embedding_json
        )
        
        if not rows:
            logfire.warning("No relevant documentation found")
            return "No relevant documentation found."
            
        # Format the results
        logfire.info("Found relevant documentation", result_count=len(rows))
        formatted_chunks = []
        for row in rows:
            metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'])
            chunk_text = f"""
# {row['title']}
URL: {row['url']}

{row['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        result = "\n\n---\n\n".join(formatted_chunks)
        logfire.info("Returning formatted documentation chunks", chunk_count=len(formatted_chunks))
        return result
        
    except Exception as e:
        logfire.error("Error retrieving documentation", error=str(e))
        return f"Error retrieving documentation: {str(e)}"

@agent.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    logfire.info("Starting list_documentation_pages")
    try:
        # Query the database directly using SQL
        logfire.info("Querying database for documentation pages")
        rows = await ctx.deps.pool.fetch(
            """
            SELECT DISTINCT url 
            FROM site_pages 
            WHERE metadata->>'source' = 'pydantic_ai_docs'
            ORDER BY url
            """
        )
        
        if not rows:
            logfire.warning("No documentation pages found")
            return []
            
        # Extract unique URLs
        urls = [row['url'] for row in rows]
        logfire.info("Retrieved documentation page URLs", url_count=len(urls))
        return urls
        
    except Exception as e:
        logfire.error("Error retrieving documentation pages", error=str(e))
        return []

@agent.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the database pool
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    logfire.info("Starting get_page_content", url=url)
    try:
        # Query the database directly using SQL
        logfire.info("Querying database for page content", url=url)
        rows = await ctx.deps.pool.fetch(
            """
            SELECT title, content, chunk_number 
            FROM site_pages 
            WHERE url = $1 
            AND metadata->>'source' = 'pydantic_ai_docs'
            ORDER BY chunk_number
            """,
            url
        )
        
        if not rows:
            logfire.warning("No content found for URL", url=url)
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = rows[0]['title'].split(' - ')[0]  # Get the main title
        logfire.info("Formatting page content", title=page_title, chunk_count=len(rows))
        formatted_content = [f"# {page_title}\nURL: {url}\n"]
        
        # Add each chunk's content
        for row in rows:
            formatted_content.append(row['content'])
            
        # Join everything together
        result = "\n\n".join(formatted_content)
        logfire.info("Returning formatted page content", content_length=len(result))
        return result
        
    except Exception as e:
        logfire.error("Error retrieving page content", error=str(e), url=url)
        return f"Error retrieving page content: {str(e)}"
    
@asynccontextmanager
async def database_connect() -> AsyncGenerator[asyncpg.Pool, None]:
    """Connect to the pgvector database."""
    logfire.info("Starting database_connect")
    # Database connection parameters from compose.yml
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'pg',
        'password': 'rag01',
        'database': 'rag'
    }
    
    try:
        # Create a connection pool
        logfire.info("Creating database connection pool", **db_config)
        pool = await asyncpg.create_pool(**db_config)
        logfire.info("Successfully connected to database")
        yield pool
    finally:
        if 'pool' in locals():
            logfire.info("Closing database connection pool")
            await pool.close()

async def run_agent(question: str):
    """Run the agent with the provided question."""
    logfire.info("Starting run_agent", question=question)
    print(f"Question: {question}")
    
    # Connect to the database and create OpenAI client
    logfire.info("Connecting to database")
    async with database_connect() as pool:
        # Create dependencies
        logfire.info("Creating dependencies")
        deps = PydanticAIDeps(
            pool=pool,
            openai_client=client
        )
        
        # Run the agent
        logfire.info("Running agent")
        response = await agent.run(question, deps=deps)
        
        # Print the response
        logfire.info("Agent completed execution", response_length=len(response.data))
        print("\nResponse:")
        print(response.data)
        
        return response.data

async def main():
    """Main function to run the agent."""
    logfire.info("Starting main")
    question = "What is Pydantic AI? Give me some usecases and examples."
    logfire.info("Calling run_agent", question=question)
    result = await run_agent(question)
    logfire.info("Finished main", result_length=len(result) if result else 0)

if __name__ == "__main__":
    logfire.info("Script started")
    asyncio.run(main())
    logfire.info("Script completed")
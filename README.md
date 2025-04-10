# LearnAgent

> 🚧 This project is still under active development and is not yet finished. Expect frequent changes and incomplete features.


## Run

- Create & activate virtual python environment

*Below commands are for macOS. Commands in Windows & Linux may vary.*

```bash
python3 -m venv venv
source venv/bin/activate
```

- Install dependencies

```bash
pip install -r requirements.txt
```

- Run the database with docker compose

```bash
docker compose up -d
```

- Make sure [Ollama](https://github.com/ollama/ollama?tab=readme-ov-file) is running

*LLM & Embedding model will be auto pulled first, if not already in Ollama*

- Create `.env`

```bash
mv .env.example .env
```

- Crawl a website

```bash
python crawl_docs.py
```

- Run RAG

```bash
python agent.py
```

**Adding new package**

- Add it into `requirements.in`
- Update requirements, `pip-compile requirements.in`
- Install package, `pip install -r requirements.txt`
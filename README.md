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

- Crawl a website

```bash
python crawl_docs.py
```

**Adding new package**

- Add it into `requirements.in`
- Update requirements, `pip-compile requirements.in`
- Install package, `pip install -r requirements.txt`
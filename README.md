# Graph RAG — Knowledge Base with Graph-Augmented Retrieval

A fully offline-capable RAG system that builds a knowledge graph from your documents and uses hybrid vector + graph search to answer questions. Works with LM Studio (local) or the Claude API.

## Features

- Ingest `.txt`, `.md`, and `.pdf` files into a persistent knowledge base
- Hybrid retrieval: vector similarity + knowledge graph traversal
- Fully offline with LM Studio, or online with Claude API (switchable via `.env`)
- MCP server — use directly from Claude Code or LM Studio
- Supabase + pgvector backend via Docker (no cloud account needed)

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Python 3.11+
- [LM Studio](https://lmstudio.ai/) (optional, for offline LLM)

## Quick Start

**Clone and run the installer — it handles everything:**
```bash
git clone <repo-url>
cd graph_rag
python install.py
```

The installer will ask 2 questions and take care of the rest:
1. Your platform (Windows / macOS / Linux) — auto-detected
2. Docker available? — installs via Docker or native PostgreSQL accordingly

**After installation:**
```bash
python main.py ingest path/to/document.pdf   # add a document
python chat.py                               # start chatting
```

### Manual setup (if preferred)

```bash
python generate_keys.py    # generate .env
docker compose up -d       # start database
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py ingest path/to/document.pdf
python chat.py
```

## LLM Provider

Set `LLM_PROVIDER` in `.env`:

| Value | Requires | Notes |
|---|---|---|
| `lmstudio` | LM Studio running locally | Fully offline |
| `claude` | `ANTHROPIC_API_KEY` | Requires internet |

## MCP Server (Claude Code integration)

Register once from the project directory:
```bash
claude mcp add graph-rag -s user -- py mcp_server.py
```

Available MCP tools:
- `ingest_document` — add a file to the knowledge base
- `retrieve_context` — search for relevant content
- `search_chunks` — raw vector similarity search
- `get_knowledge_base_stats` — show document/chunk/graph counts
- `build_knowledge_graph` — rebuild graph from all stored documents

## Architecture

```
Documents → Chunker → Embeddings → PostgreSQL (pgvector)
                  ↘ Entity Extractor → Knowledge Graph
Query → Vector Search + Graph Traversal → Context → LLM → Answer
```

## Services & Ports

| Service | URL |
|---|---|
| PostgreSQL | `localhost:5432` |
| Supabase Studio | `http://localhost:3001` |
| PostgREST API | `http://localhost:3000` |
| MCP HTTP server | `http://localhost:8080/mcp` |
| LM Studio | `http://localhost:1234/v1` |

## Documentation

- [SETUP_GUIDE.txt](SETUP_GUIDE.txt) — detailed setup walkthrough and architecture notes
- [COMMANDS.txt](COMMANDS.txt) — full CLI reference

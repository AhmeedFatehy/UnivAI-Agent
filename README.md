# UnivAI Agent — the RAG service

The retrieval brain of **UnivAI ("Jamieh")**: it indexes the uploaded textbook
and answers every "what does the book say" question the rest of the system
asks — over MCP (streamable-http).

## Run it

```bash
uv sync
uv run python mcp_server.py      # http://localhost:8000/mcp
```

Needs **Qdrant on :6333** (the UnivAI repo's `make up` starts it).

## MCP tools it exposes

| Tool | Does |
|---|---|
| `ingest_file` | index a document by absolute file path |
| `retrieve_context` | hybrid retrieval + rerank for a query |
| `list_documents` | what a user has indexed |
| `remove_document` | delete one document's chunks |

## Where to look

| Folder / file | What |
|---|---|
| `mcp_server.py` | the MCP entry point the other services call |
| `document_processing/` | parsing + chunking |
| `vector_store/` | embedding + Qdrant indexing |
| `retrieval/` | search + reranking |
| `evaluation/` | retrieval quality experiments |
| `agent.py`, `app.py` | the conversational agent + its own API |

## Consumed by

- the UnivAI app's upload flow (`ingest_file`, and cleanup on book replacement)
- the live-lecture voice agent (`retrieve_context` for raised-hand questions)

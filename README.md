# UnivAI Agent — the RAG service

The retrieval brain of **UnivAI ("Jamieh")**: it indexes the uploaded textbook
and answers every "what does the book say" question the rest of the system
asks — over MCP (streamable-http).

## Standalone development

The standalone path uses a project-authored Markdown fixture, a deterministic
token-overlap store, and recorded course material. It does not use Qdrant,
Ollama, downloaded models, PostgreSQL, Slidev, LiveKit, or voice models.

```powershell
uv sync
$env:UNIVAI_MODE="standalone"
uv run python standalone.py smoke
uv run python standalone.py generate
uv run python standalone.py reset
uv run python -m unittest discover -s tests -v
uv run python mcp_server.py
```

The same commands work in Linux shells with
`UNIVAI_MODE=standalone uv run python ...`. Generated files are written under
`.standalone/output/week-N/`. The fixture learner is `standalone-student`; a
known query is `How does tenant isolation protect learners?`.

Standalone is explicit and is rejected when `UNIVAI_ENV=production`. Its
token-overlap search is deterministic development behaviour, not a quality
test of production embeddings or reranking.

## Integrated mode

```bash
uv sync
$env:UNIVAI_MODE="integrated"
uv run python mcp_server.py      # http://localhost:8000/mcp
```

Needs **Qdrant on :6333** (the UnivAI repo's `make up` starts it).
The main repository keeps invoking generation with:

```text
python UnivAI-Agent/generation/lecture_gen.py <absolute_pdf_path> <book_id>
```

Generation first discovers grounded chapter boundaries and stores the semester
plan and every generated week artifact in PostgreSQL. Normal chapters take one
week, adjacent small chapters may share a week (never more than three), and a
large chapter may take two weeks. The number of weeks is not fixed.

Set `UNIVAI_INTEGRATION_ROOT` only when the Agent is not located directly
inside the main checkout. Integrated mode fails clearly if the parent shared
services are missing and never falls back to fixtures.

## MCP tools it exposes

| Tool | Does |
|---|---|
| `ingest_file` | index a document by absolute file path |
| `retrieve_context` | hybrid retrieval + rerank for a query |
| `list_documents` | what a user has indexed |
| `remove_document` | delete one document's chunks |
| `server_info` | mode and stable tool summary |

## Where to look

| Folder / file | What |
|---|---|
| `mcp_server.py` | the MCP entry point the other services call |
| `generation/` | course generation: `lecture_gen.py` turns the indexed book into slides + narration + quiz JSON per week (Brain makes JSON, the other caves eat it) |
| `document_processing/` | parsing + chunking |
| `cache/` | content-addressed artifact cache + tenant authorization |
| `vector_store/` | embedding + Qdrant indexing |
| `retrieval/` | search + reranking |
| `evaluation/` | retrieval quality experiments |
| `prompts/` | versioned system prompts and the operation-to-prompt registry |
| `agent.py`, `app.py` | the conversational agent + its own API |

## Consumed by

- the UnivAI app's upload flow (`ingest_file`, and cleanup on book replacement)
- the live-lecture voice agent (`retrieve_context` for raised-hand questions)

Generated presentations use the bounded visual vocabulary documented in
[`docs/slide-generation-contract.md`](docs/slide-generation-contract.md). The
model supplies semantic JSON; repository-owned code compiles it safely to
Slidev.

## Contracts and safety

`contracts.py` owns the MCP tool list, course-size table, and script/quiz
validation used by the smoke command. Standalone ingest accepts only
project-authored text/Markdown beneath `fixtures/` or the configured
`.standalone/uploads/` directory. Reset refuses paths outside this repository.

### Content-addressed RAG cache

Identical bytes are parsed, chunked and embedded exactly once per pipeline
version, then safely reused by every tenant that uploads them:

- `cache/content_identity.py` hashes the raw file bytes (SHA-256) and
  fingerprints the whole pipeline (parsers, splitter, chunk size/overlap,
  embedding models, metadata schema, document type).
- `cache/artifact_registry.py` stores one immutable artifact per
  `content_hash:fingerprint` key with an atomic `building → ready | failed |
  corrupt` state machine, one-writer builds across concurrent uploads, and
  re-verification of byte length + hash before any reuse. It is backed by
  `.cache/artifacts/` (override with `UNIVAI_ARTIFACT_CACHE_ROOT`).
- `cache/authorization.py` gives every tenant an independent, revocable grant
  on a shared artifact. Retrieval and citation resolution refuse any document
  without an active grant (`REFUSAL_NO_GRANT`), and deleting the last reference
  removes the artifact.

The cache never stores filenames, tenant identity or raw book text; telemetry
is an append-only `events.jsonl` carrying only keys, fingerprints, sizes and
outcomes.

Real model/provider testing is opt-in through integrated mode. Model downloads,
Qdrant failures, and Ollama failures therefore cannot be mistaken for
standalone success. Scanned PDFs still require OCR before integrated
generation.

This directory is a Git submodule in the main UnivAI repository. Commit and
merge Agent changes here first, then update the main repository's gitlink.
Local file changes inside this directory are not included automatically in a
main-repository commit.

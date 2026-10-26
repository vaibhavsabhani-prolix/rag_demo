# Patent Semantic Search

Patent Semantic Search is a React and FastAPI application backed by Qdrant.
It indexes patent text into vector collections and exposes a streaming search
UI with query understanding, metadata filtering, verification, reranking, and
final patent scoring. Search history is stored in PostgreSQL.

## Architecture

```text
Patent .txt files
        |
        v
  app.ingest
  parse -> chunk -> embed -> write
        |
        v
      Qdrant
        |
        +----------------------+
        |                      |
        v                      v
  FastAPI /api             React/Vite UI
        |                      |
        +---- PostgreSQL <-----+
              search history
```

The browser calls `/api` endpoints. Search endpoints return newline-delimited
JSON (NDJSON), so the UI can render phase progress while the pipeline is
running. In the production-style Docker image, FastAPI serves the built React
files as well.

## Requirements

- Python 3.12
- Node.js 24 and npm (only needed for local frontend development)
- Docker and Docker Compose
- A running Qdrant instance
- A running embedding service, query-LLM service, and reranker service

The remote model URLs and model names are configured in
[app/config.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/config.py).
Change those values or provide a configuration mechanism appropriate for your
environment before starting the application.

## Quick start with Docker

Start the production-style stack:

```bash
docker compose -f docker-compose.yml up -d --build
```

Open <http://localhost:8000>. The stack starts:

| Service | Address | Purpose |
| --- | --- | --- |
| `web` | <http://localhost:8000> | FastAPI API and built React UI |
| `qdrant` | <http://localhost:6333> | Vector database |
| `postgres` | `localhost:5434` | Search history database |
| `adminer` | <http://localhost:8080> | PostgreSQL browser |

Adminer uses server `postgres`, user `patent`, password `patent`, and database
`patent_search`.

For hot reload during development:

```bash
docker compose up
```

This uses [docker-compose.override.yml](/Users/hepinprolix/Desktop/demo/rag_demo/docker-compose.override.yml):
FastAPI reloads on backend changes and Vite serves the UI at
<http://localhost:5173>. The Vite server proxies `/api` to the `web` service.

## Local development

Create a virtual environment and install backend dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Start Qdrant and PostgreSQL with Compose, then run the API:

```bash
docker compose up -d qdrant postgres
./run_api.sh
```

In another terminal, install and start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

The local Vite proxy targets `http://localhost:8000` by default. Override it
with `API_PROXY_TARGET` when the API is elsewhere.

## Indexing patents

Patent files are read from `patents/1` by default. Each `*.txt` file is parsed,
split into validated token-window chunks, embedded in batches, and written to
two Qdrant collections:

- `patent_chunks_1024`: chunk vectors used for retrieval
- `patents_metadata_1024`: patent-level metadata

Run indexing from the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m app.ingest
```

The ingest pipeline uses bounded queues and worker pools for parsing/chunking,
embedding, and Qdrant insertion. Completed filenames are appended to
`data/ingest_progress_1024.log`, so rerunning the command skips completed
files. By default, Qdrant indexing is paused during ingestion and built after
the writes finish. To index while ingesting:

```bash
INGEST_PAUSE_INDEXING=0 PYTHONPATH=. .venv/bin/python -m app.ingest
```

If an ingest was interrupted after writes completed but before the Qdrant
index finished, resume or watch index creation with:

```bash
PYTHONPATH=. .venv/bin/python -m app.scripts.build_index
```

## Search flow

The standard search endpoint is `POST /api/search`. The backend emits:

1. **Query Understanding** — the remote query LLM normalizes the query and
   extracts metadata constraints; results may come from the in-memory cache.
2. **Candidate Vector Retrieval** — Qdrant retrieves up to the configured
   chunk limit.
3. **Metadata Filtering** — candidates are filtered against the parsed
   constraints.
4. **Semantic Relationship Verification** — an LLM checks relationship and
   requirement support.
5. **BGE Cross-Encoder Reranking** — the reranker scores candidate chunks and
   supplies sentence-level highlighting signals.
6. **Final Patent Scoring** — retrieval, reranker, relationship, and
   requirement signals are combined; patents meeting the final score threshold
   are returned.

The endpoint first emits a `start` event, then one `phase` event per completed
phase, followed by `done` or `error`. Search history is written before the
terminal event when PostgreSQL is available.

## Other application flows

- `/api/compare` parses a query once and runs phases 2–6 sequentially for each
  selected collection.
- `/api/keyfeature-search` extracts invention key features, embeds them, and
  runs the key-feature retrieval/reranking pipeline for one collection.
- `/api/keyfeature-compare` extracts and embeds features once, then evaluates
  them across selected collections.
- `/api/keyfeature-extract` returns extracted features without searching.
- `/api/collections` lists searchable Qdrant collection pairs.
- `/api/collections/stats` and `/api/collections/patent/{patent_id}` expose
  collection and patent inspection data.
- `/api/history` provides persisted searches and phase details.
- `/api/cache` reports or clears query-understanding cache entries.
- `/api/settings/ui` persists UI preferences.

## Frontend routes

| Route | Feature |
| --- | --- |
| `/` | Standard semantic search and live pipeline |
| `/features` | Key-feature search |
| `/feature-compare` | Key-feature comparison |
| `/compare` | Search comparison across collections |
| `/collections` | Collection statistics and patent lookup |
| `/history` | Search history |
| `/history/:id` | Stored search details and phases |

## Configuration

Backend settings live in [app/config.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/config.py).
Important values include:

- `QDRANT_HOST`, `QDRANT_PORT`, and `QDRANT_GRPC_PORT`
- `DATABASE_URL`
- remote embedding, query-LLM, and reranker URLs/models
- `PATENT_DIRECTORY`
- chunking, batching, concurrency, and indexing settings
- final score threshold and signal weights

In Docker, Compose supplies service hostnames (`qdrant` and `postgres`) to the
`web` container. Do not commit credentials or private service endpoints.

## Validation

Backend tests are in `app/_tests_`. Run the focused unit tests or the full test
set with the repository's configured Python test runner. Frontend checks:

```bash
cd frontend
npm run build
npm run lint
```

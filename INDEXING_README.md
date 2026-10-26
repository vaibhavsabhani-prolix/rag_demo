# Patent indexing

This guide documents the indexing path implemented in
[app/ingest.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/ingest.py).
Indexing is a batch process that turns patent text files into searchable Qdrant
vectors and patent metadata.

## Data flow

```text
patents/1/*.txt
    |
    v
scan files and skip completed filenames
    |
    v
parse patent metadata and text
    |
    v
split into validated token-window chunks
    |
    v
embed chunk windows through the remote embedding API
    |
    v
write vectors to patent_chunks_1024
write completed patent metadata to patents_metadata_1024
    |
    v
build Qdrant HNSW index
```

The default directory and collection names are defined in
[app/config.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/config.py).
The configured embedding vector size is 1024.

## Prerequisites

1. Start Qdrant on port 6333, or set `QDRANT_HOST` and `QDRANT_PORT`.
2. Make the embedding service configured by `EMBEDDING_REMOTE_BASE_URL`
   reachable.
3. Place patent files with a `.txt` extension under `patents/1`.
4. Install the dependencies from `requirements.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
docker compose up -d qdrant
```

## Run indexing

From the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m app.ingest
```

`app.ingest` creates the configured Qdrant collections if needed, scans the
input directory, and reports counts for patents, chunks, inserted vectors,
and failures.

## Pipeline stages

The implementation uses bounded queues to apply backpressure:

1. **Parse and chunk**: a thread pool loads each patent and creates validated
   chunks. Each worker owns its chunker state.
2. **Chunk dispatch**: chunks are buffered until the configured prefetch
   capacity and then cut into embedding windows.
3. **Embedding**: `EMBED_CONCURRENT_REQUESTS` workers send windows of at most
   `EMBED_BATCH_SIZE` chunks to the remote embedding service.
4. **Vector dispatch**: completed windows are combined into Qdrant insertion
   batches of up to `BATCH_SIZE` chunks.
5. **Qdrant insertion**: `INSERT_WORKERS` workers write chunk vectors and
   patent metadata. A patent is checkpointed only after all of its chunks are
   written.

Relevant tuning values are `CHUNK_QUEUE_CAPACITY`, `EMBED_BATCH_SIZE`,
`EMBED_CONCURRENT_REQUESTS`, `BATCH_SIZE`, and `INSERT_WORKERS`.

## Resume behavior and failures

After a patent's metadata and every chunk have been written, its source
filename is appended to `data/ingest_progress_1024.log`. A later run reads
this append-only file and skips those filenames. Files that fail parsing,
embedding, or writing are reported and are not treated as completed.

The progress file is a local checkpoint, not a replacement for validating the
Qdrant collections. If it is removed, the files are eligible for processing
again; Qdrant upserts must still be checked for duplicate or stale data.

## Qdrant index creation

By default, `INGEST_PAUSE_INDEXING=1`. In this mode ingestion pauses Qdrant
indexing while vectors are inserted, resumes it in a `finally` block, and
waits until the chunks collection is green.

To allow Qdrant to index continuously:

```bash
INGEST_PAUSE_INDEXING=0 PYTHONPATH=. .venv/bin/python -m app.ingest
```

If indexing is still running after an interrupted command, watch it with:

```bash
PYTHONPATH=. .venv/bin/python -m app.scripts.build_index
```

That command resumes indexing and prints progress until
`patent_chunks_1024` is green. It should not be run concurrently with an
active ingest.

## Inspect indexed patents

Use the inspection script to query the metadata collection:

```bash
PYTHONPATH=. .venv/bin/python -m app.scripts.show_indexed_patents
```

The web UI also exposes collection statistics and patent lookup under
`/collections`, backed by `/api/collections/stats` and
`/api/collections/patent/{patent_id}`.

## Configuration reference

| Setting | Default | Role |
| --- | --- | --- |
| `PATENT_DIRECTORY` | `patents/1` | Source directory |
| `CHUNKS_COLLECTION_NAME` | `patent_chunks_1024` | Chunk vector collection |
| `PATENTS_COLLECTION_NAME` | `patents_metadata_1024` | Metadata collection |
| `MAX_CHUNK_TOKENS` | `1024` | Chunk token limit |
| `EMBED_BATCH_SIZE` | `128` | Remote embedding window |
| `BATCH_SIZE` | `512` | Qdrant insertion batch |
| `EMBED_CONCURRENT_REQUESTS` | `5` | Embedding worker count |
| `INSERT_WORKERS` | `4` | Qdrant insertion worker count |
| `INGEST_PROGRESS_FILE` | `data/ingest_progress_1024.log` | Completion checkpoint |
| `INGEST_PAUSE_INDEXING` | `1` | Pause HNSW indexing during ingest |

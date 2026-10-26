# Search flow

The standard search implementation is split between the API adapter in
[app/api/main.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/api/main.py),
the orchestration in
[app/semantic_search.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/semantic_search.py),
and the phase implementations under `app/query_understanding`,
`app/retrieval`, `app/verification`, `app/reranking`, and `app/scoring`.

## Request lifecycle

```text
React SearchForm
    |
    | POST /api/search
    v
resolve collection -> save history start
    |
    v
stream NDJSON from a worker thread
    |
    +-- start
    +-- phase 1 ... phase 6
    +-- done or error
    |
    v
save completed phases and return to the UI
```

The frontend validates every event with Zod before updating the pipeline views.
The API uses the selected collection, or the ingestion collection
`patent_chunks_1024` when no collection is supplied.

## Six standard phases

`SearchPipeline.run()` executes these phases in order:

### 1. Query understanding

`QueryUnderstandingEngine` sends the user's query to the configured query LLM.
It produces a normalized semantic query, relationship and requirement
hypotheses, and metadata filters. Results can be served from the in-memory
query cache.

### 2. Candidate vector retrieval

`CandidateRetriever` embeds the parsed query and retrieves up to
`RETRIEVAL_TOP_K_CHUNKS` chunks from the selected Qdrant collection.

### 3. Metadata filtering

`filter_candidates` applies the parsed metadata constraints. Metadata-only
queries can take a specialized path that does not require the normal semantic
signals.

### 4. Semantic verification

`RelationshipVerifier` evaluates whether candidate evidence supports the
query's relationship and requirement hypotheses. Verification thresholds are
configured in `app/config.py`.

### 5. BGE cross-encoder reranking

`BGEReranker` scores candidate text against the parsed query using the remote
BGE reranker. The phase also supplies sentence-level scores used to highlight
matching evidence in the UI.

### 6. Final scoring

`FinalScorer` combines retrieval, reranker, relationship, and requirement
signals using the configured weights. Results with a final score below
`FINAL_SCORE_THRESHOLD` are excluded; there is no fixed final top-K cap.

## Streaming event format

`POST /api/search` returns `application/x-ndjson`. A typical stream contains:

```json
{"type":"start","query":"...","collection":"...","cache_hit":false,"search_id":123}
{"type":"phase","phase":1,"name":"Query Understanding","elapsed_ms":12.3,"data":{}}
{"type":"phase","phase":6,"name":"Final Patent Scoring","elapsed_ms":45.6,"data":{}}
{"type":"done","elapsed_ms":1234.5}
```

Failures are sent as an `error` event with the exception type and message.
Unknown collections fail before streaming with a normal HTTP error.

## Compare flow

`POST /api/compare` accepts a query and a list of collection names. Query
understanding runs once with cache disabled. Each collection then runs phases
2–6 from the same parsed query, in sequence. Events include the collection
name, collection start/done markers, per-phase timing, and memory samples.
Failure of one collection is reported as `collection_error`; other selected
collections continue.

## Key-feature flows

The key-feature pipeline is separate from the six-phase semantic pipeline:

- `POST /api/keyfeature-extract` extracts structured key features from the
  problem, invention title, and invention details.
- `POST /api/keyfeature-search` extracts/embeds features, retrieves matching
  patent chunks, reranks feature evidence, and streams phase events for one
  collection.
- `POST /api/keyfeature-compare` extracts and embeds features once, then
  evaluates the same features across each selected collection.

The compare endpoint can reuse caller-provided features, avoiding repeated
extraction.

## Supporting endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/config` | Exposes pipeline models, limits, weights, and patent links |
| `GET /api/collections` | Lists valid searchable collection pairs |
| `GET /api/cache` | Reports query cache statistics |
| `DELETE /api/cache` | Clears the query cache |
| `GET /api/history` | Lists persisted searches with pagination and query filtering |
| `GET /api/history/{id}` | Returns one stored search |
| `GET /api/history/{id}/phases` | Returns stored phase data |
| `DELETE /api/history/{id}` | Deletes one stored search |
| `DELETE /api/history` | Clears search history |

History persistence is best effort: a PostgreSQL failure is logged and does
not prevent the search stream from running.

## Configuration

Search settings are in
[app/config.py](/Users/hepinprolix/Desktop/demo/rag_demo/app/config.py).
The most important controls are:

- `RETRIEVAL_TOP_K_CHUNKS`
- `RERANK_BATCH_SIZE`, `RERANK_CONCURRENT_REQUESTS`, and reranker context limits
- `VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD`
- `VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD`
- `FINAL_SCORE_THRESHOLD`
- `FINAL_WEIGHT_RELATIONSHIP`, `FINAL_WEIGHT_REQUIREMENT`,
  `FINAL_WEIGHT_RERANKER`, and `FINAL_WEIGHT_RETRIEVAL`
- `HIGHLIGHT_SENTENCE_THRESHOLD` and
  `HIGHLIGHT_SENTENCE_STRONG_THRESHOLD`

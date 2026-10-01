# Patent Semantic Search System

A production-ready Patent Retrieval & Semantic Search system built with **Python**, **Qdrant Vector DB**, a **remote Qwen3 embedding server**, an **LLM-based Query Understanding layer**, **multi-view dynamic retrieval**, **metadata filtering**, a **cross-encoder relationship/requirement verification stage**, a **BGE cross-encoder reranker**, and **weighted multi-signal final scoring with query-match highlighting**.

The pipeline handles end-to-end processing of complex technical patent documents: from raw document parsing and token-window chunking, through validation, remote vector embedding, and concurrent batch indexing — to natural-language query understanding, multi-view semantic candidate retrieval, metadata filtering, bounded evidence retrieval, cross-encoder relationship/requirement verification, BGE reranking with highlighting, and deterministic multi-signal final scoring. A React + FastAPI web app (with search history in PostgreSQL) and CLI tools are included for search, collection comparison, and index inspection.

---

## 📐 System Architecture & Flow

```mermaid
flowchart TD
    subgraph Ingestion_Pipeline ["1. Ingestion & Indexing Pipeline (app/ingest.py)"]
        A["Raw Patent Files (.txt + .json)"] --> B["Patent Parser (app/parser.py)"]
        B --> C["Patent Document Object"]
        C --> D["Section Detector (app/chunking/section_detector.py)\nmatches lines against a known-heading whitelist"]
        D --> E["Token Window Chunker (app/chunking/token_window_chunker.py)\ntokenizes each section once, cuts MAX_CHUNK_TOKENS windows,\nadjusts boundaries to paragraph/sentence/word/char"]
        E --> F["Chunk Validator (app/chunking/chunk_validator.py)\nrejects empty/whitespace/low-info/degenerate-overlap chunks"]
        F --> G["Embedder (app/embedder.py)\nremote vLLM embedding server, batched + concurrent HTTP"]
        G --> H1["patent_chunks_&lt;name&gt; collection (vectors + chunk payload)"]
        C --> H2["patents_metadata_&lt;name&gt; collection (metadata only, no vectors)"]
    end

    subgraph Search_Pipeline ["2. Seven-Phase Search Pipeline (app/semantic_search.py SearchPipeline)"]
        J["User Search Query"] --> P1["Phase 1 - Query Understanding\n(app/query_understanding/engine.py)\n-> ParsedQuery: semantic_query, concepts, relationships,\nattributes, requirements, constraints, exclusions,\nmetadata_filters, is_metadata_only"]

        P1 -->|"is_metadata_only"| MO["CandidateRetriever._retrieve_metadata_only\nscrolls patents_metadata_* collection,\nPython-verifies filters, NO vector search"]
        P1 -->|"semantic query"| P2["Phase 2 - Candidate Retrieval\n(app/retrieval/retriever.py)\n3 dynamic views (original / semantic / structured)\nembedded once each, metadata pre-filter narrows the\nQdrant search (falls back to unrestricted on 0 matches),\nquery_points per view (RETRIEVAL_TOP_K_PER_VIEW each),\ndeduped + grouped by patent_id, bounded to\nPATENT_CANDIDATE_TOP_K patents by best chunk score"]
        MO --> P3
        P2 --> P3["Phase 3 - Metadata Filtering\n(app/retrieval/metadata_filter.py)\nstrict AND over parsed_query.metadata_filters,\ndate/code/text-aware comparison, per-filter diagnostics"]

        P3 --> P4["Phase 4 - Bounded Evidence Retrieval\n(app/retrieval/evidence_retriever.py)\ndeterministic evidence query (semantic_query +\nrelationships + requirements + concepts), ONE vector\nsearch restricted to surviving patent_ids\n(EVIDENCE_GLOBAL_TOP_K_CHUNKS global cap, not\nper-patent) + EVIDENCE_NEIGHBOR_CHUNKS neighbors\nfetched in one batched scroll"]

        P4 --> P5["Phase 5 - Relationship & Requirement Verification\n(app/verification/verifier.py)\nBGE cross-encoder scores every relationship/requirement\nhypothesis against every evidence chunk + deterministic\nword-proximity check; produces per-patent coverage\nratios - nothing is dropped here"]

        P5 --> P6["Phase 6 - BGE Cross-Encoder Reranking\n(app/reranking/reranker.py)\none deterministic reranking query, token-budgeted chunks\nscored in batched HTTP calls; patent score = MAX across\nits own chunks; sentences scored in the same batch for\nquery-match highlighting (app/highlighting/)"]

        P6 --> P7["Phase 7 - Final Scoring & Result Selection\n(app/scoring/scorer.py)\nweighted composite: 0.45*relationship_coverage +\n0.25*requirement_coverage + 0.20*best_reranker_score +\n0.10*retrieval_score, 0-10 scale; filtered by\nFINAL_SCORE_THRESHOLD, sorted descending\n(metadata-only: unscored, unranked, all returned)"]

        P7 --> R["React UI (frontend/) + FastAPI (app/api/main.py)\nstreams one NDJSON event per phase / CLI"]
    end
```

---

## ⚙️ End-to-End Process Breakdown

### Stage 1: Patent Data Reading & Parsing (`app/parser.py`)
* **Input Data Structure**:
  * Raw Patent Text (`.txt`): Contains full patent text with section headers (Abstract, Background, Detailed Description, Claims).
  * Patent Metadata (`.json`): Contains structured metadata like Patent ID, Title, Filing Date, Classification codes, Assignee/Applicant, and Inventors.
* **Process**:
  1. `PatentParser.load_patent(txt_path)` reads both text and corresponding `.json` metadata side-by-side.
  2. Constructs a unified `PatentDocument` dataclass containing `patent_id`, `text`, and `metadata`.

---

### Stage 2: Section Detection & Token-Window Chunking (`app/chunker.py` + `app/chunking/*`)
Patents require strict structural isolation — chunks must never mix contents across document sections.

* **Step 2.1 - Section Detector (`section_detector.py`)**:
  * Verifies each candidate heading line against a known-heading whitelist (`KNOWN_SECTION_HEADINGS` in `app/chunking/known_headings.py`), bounded by `MAX_HEADING_LENGTH`, `MAX_HEADING_WORDS`, and `ALLCAPS_MIN_ALPHA` — not generic all-caps/colon heuristics.
  * Yields isolated `Section` objects. Each section is processed as an independent stream.

* **Step 2.2 - Token Window Chunker (`token_window_chunker.py`)**:
  * Tokenizes each section's content **once** (`TokenCounter`, HuggingFace tokenizer with an LRU cache) to get token IDs and character offsets.
  * Cuts target windows bounded by `MAX_CHUNK_TOKENS` (default: 512), then locally adjusts each boundary backward to the nearest safe textual boundary in priority order: paragraph break → sentence boundary (multilingual, abbreviation-aware) → word boundary → clean UTF-8 character boundary.
  * Slices the original text using the tokenizer's own character offsets and reports exact token counts without re-tokenizing.

---

### Stage 3: Chunk Quality Validation & Filtering (`app/chunking/chunk_validator.py`)
To prevent indexing low-quality vector noise into Qdrant, every chunk passes through validation. Core checks (empty / whitespace-only / duplicate) always run; the rest are config-driven and all currently enabled:

1. **Whitespace Normalization** (`VALIDATOR_NORMALIZE_WHITESPACE`): Collapses redundant tabs, double spaces, and newline padding.
2. **Heading-Only Rejection** (`VALIDATOR_REJECT_HEADING_ONLY`): Filters out isolated headings without body content, up to `VALIDATOR_HEADING_ONLY_MAX_WORDS` words.
3. **Low-Information Filtering** (`VALIDATOR_REJECT_LOW_INFO`): Rejects chunks whose alphabetic-character ratio falls below `VALIDATOR_LOW_INFO_THRESHOLD` (0.30), eliminating table artifacts, binary noise, and separator lines.
4. **Degenerate Overlap Prevention** (`VALIDATOR_REJECT_DEGENERATE_OVERLAP`): Rejects near-duplicate chunks whose unique-token ratio against the previous chunk falls below `VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD` (0.9).

The validator never rejects a chunk purely for being small — every section's content is legitimate and must be preserved.

---

### Stage 4: Vector Embedding Generation (`app/embedder.py`)
* **Model**: `Qwen/Qwen3-Embedding-0.6B`, served by a **remote vLLM server** behind an OpenAI-compatible `/embeddings` endpoint (`EMBEDDING_REMOTE_BASE_URL`) — not run in-process.
* **Output Dimension**: 1024-dimensional dense vectors (`VECTOR_SIZE = 1024`).
* **Throughput**: `EMBED_BATCH_SIZE` texts per HTTP request (large batches amortize round-trip latency and keep the remote GPU busy); `EMBED_CONCURRENT_REQUESTS` requests in flight at once via a thread pool, so the next batch is already in transit while the current one computes; a persistent `requests.Session` reuses TCP connections.

---

### Stage 5: Vector DB Indexing & Storage (`app/qdrant_db.py` & `app/ingest.py`)
* **Vector Store**: **Qdrant** running via Docker on port `6333`.
* **Collection Discovery**: search collections are discovered by naming convention — every `patent_chunks_<name>` with a matching `patents_metadata_<name>` is listed as a searchable collection `<name>` in the UI (`QdrantDB.list_search_collections`). Ingestion writes to the pair named by `CHUNKS_COLLECTION_NAME` / `PATENTS_COLLECTION_NAME` (currently the `512` suffix, i.e. `MAX_CHUNK_TOKENS`).
  * `patent_chunks_<name>` — one point per chunk, with its embedding vector and full chunk payload. Searched.
  * `patents_metadata_<name>` — one point per patent, metadata only, no vectors. Looked up by `patent_id` for display and metadata filtering; never searched by vector.
* **Distance Metric**: Cosine Similarity.
* **Concurrent Ingestion Pipeline** (`app/ingest.py`) — five stages, each with its own worker pool, connected by bounded queues so a slow stage backs up the one feeding it rather than stalling the whole run:
  1. Parallel parsing/chunking workers parse + chunk patent files into a patent-level `parsed_queue`.
  2. A chunk dispatcher buffers raw chunks — the first `CHUNK_QUEUE_CAPACITY` chunks accumulate before the first embedding window is cut, then one window is cut every time `>= EMBED_BATCH_SIZE` chunks are buffered.
  3. `EMBED_CONCURRENT_REQUESTS` long-lived embedding workers each embed one window and loop straight back for the next ready one.
  4. A vector dispatcher buffers embedded windows and cuts an insertion batch once `BATCH_SIZE` chunks are on hand.
  5. `INSERT_WORKERS` insert workers each upsert one batch (chunk vectors + patent metadata) to Qdrant and checkpoint the patents just committed.
  * `INGEST_PROGRESS_FILE` is an append-only log of fully-committed patent filenames; re-running the same ingest command resumes instead of reprocessing from the start.
  * `INGEST_PAUSE_INDEXING` (default on): pauses HNSW index building on the chunks collection during ingest and builds it once at the end (`app/scripts/build_index.py` can resume this if an ingest crashed mid-run), since Qdrant otherwise rebuilds the index on every segment merge, competing with ingest for CPU/disk.
* **Chunk Payload Attributes**: `patent_id`, `section`, `text`, `chunk_id`, `document_chunk_index`, `token_count`, `word_count` (see `app/models/patent_chunk.py`).

---

### Stage 6: Query Understanding (`app/query_understanding/engine.py`, Search Phase 1)
A remote LLM (`QUERY_LLM_REMOTE_BASE_URL` / `QUERY_LLM_REMOTE_MODEL`) parses the raw natural-language query into a structured `ParsedQuery` (`app/models/parsed_query.py`), with an LRU result cache (`QUERY_CACHE_SIZE`) keyed on the query text:

* `original_query`, `semantic_query` — the raw query and an intent-preserving natural-language rewrite.
* `concepts` — distinct technical concepts/entities extracted from the query.
* `relationships` — directed `SemanticRelationship(subject, relation, object, context)` triples.
* `attributes` — `ConceptAttribute(concept, name, value)` properties modifying a concept.
* `requirements` / `constraints` / `exclusions` — free-text functional requirements, numerical/physical constraints, and negative requirements (note: `exclusions` is parsed but not yet consumed anywhere in the search path).
* `metadata_filters` — structured `MetadataFilter(field, operator, value, raw_field)` constraints mapped to the field allowlist (`app/query_understanding/field_mapping.py`).
* `is_metadata_only` — true only when the query is exclusively metadata constraints with zero semantic/technical content; routes Phase 2 to the metadata-only fast path.

A truncated/malformed LLM completion is recovered by `_repair_truncated_json`, which walks the output tracking bracket/string nesting and closes it at the last cleanly-ended value instead of discarding the whole response — every `ParsedQuery` field has a safe empty default, so a partial object still validates.

---

### Stage 7: Candidate Retrieval (`app/retrieval/retriever.py` + `app/retrieval/views.py`, Search Phase 2)

**Metadata-only branch** (`is_metadata_only=True`): `_retrieve_metadata_only` builds a Qdrant filter, scrolls the `patents_metadata_*` collection, verifies every candidate against `parsed_query.metadata_filters` in Python (`_fetch_matching_patents`), and truncates to `PATENT_CANDIDATE_TOP_K` — no embedding call and no vector search at all.

**Semantic branch**:
1. `build_retrieval_views` deterministically builds up to three query views with no LLM call: `original` (raw query), `semantic` (the Query Understanding rewrite), and `structured` (a compact synthesis of concepts, relationships, attributes, and requirements).
2. Each distinct view text is embedded once in a single batched call.
3. If `metadata_filters` are present, matching patent IDs are fetched first and used to restrict the Qdrant vector search via `MatchAny(patent_id in [...])`. If that pre-filter matches **zero** patents (which is as likely to be an extraction/mapping gap as a genuine no-match), the search falls back to unrestricted and lets Phase 3 enforce the filters on whatever candidates come back.
4. Each view runs its own `query_points` call against `patent_chunks_*` with `limit=RETRIEVAL_TOP_K_PER_VIEW` (default 500).
5. Chunks are deduplicated by point ID across views (merging `matched_views` and keeping the max score), grouped by `patent_id`, and each patent's `retrieval_score` is its single best chunk's score. Candidates are sorted by that score and bounded to `PATENT_CANDIDATE_TOP_K` (default 300).
6. Surviving candidates' patent metadata is batch-fetched from `patents_metadata_*` for display and downstream filtering.

---

### Stage 8: Metadata Filtering (`app/retrieval/metadata_filter.py`, Search Phase 3)
Deterministic, strict-AND enforcement of `parsed_query.metadata_filters` against each candidate's metadata (`matches_all_metadata_filters`), with per-filter pass/fail diagnostics surfaced to the UI:

* **Field-type-aware comparison**: dates/years (`_compare_dates`, mixed year-vs-full-date granularity handled explicitly), classification/jurisdiction codes (`_compare_code`, exact/prefix/substring), and general text (`_compare_text`, equality or substring).
* **Missing metadata**: a filter on a field the patent doesn't have fails the patent (passes only for a `!=` operator).
* **Unknown fields**: a field with no resolvable payload key never vetoes a patent — this prevents an unmapped field from silently dropping an otherwise-matching result.
* A query with no `metadata_filters` passes every candidate through unchanged.

---

### Stage 9: Bounded Evidence Retrieval (`app/retrieval/evidence_retriever.py`, Search Phase 4)
For the patents that survived Phase 3, gathers the chunk text needed for verification and reranking — **not** an unbounded per-patent fetch:

1. `build_evidence_query` deterministically combines `semantic_query`, `relationships`, `requirements`, and `concepts` into one compact query text (no LLM call).
2. Phase 2's already-held chunk text is seeded into the evidence pool first.
3. One additional vector search embeds that evidence query and runs `query_points` restricted to the surviving patent IDs (`MatchAny`), with `limit=EVIDENCE_GLOBAL_TOP_K_CHUNKS` (default 1000) — a **global** cap shared across the whole candidate batch, not a per-patent cap.
4. For every chunk now on hand, up to `EVIDENCE_NEIGHBOR_CHUNKS` (default 1) adjacent chunk indices on each side are fetched in a single batched Qdrant scroll, scored at `0.95 ×` the nearest anchor chunk's score, and merged in.
5. Chunks are deduplicated per patent and ordered (direct matches before neighbors, by score); every matched chunk and its neighbors are kept — there is no per-patent truncation at this stage.

---

### Stage 10: Relationship & Requirement Verification (`app/verification/verifier.py`, Search Phase 5)
A fast (sub-second), GPU-accelerated **soft coverage** check — it scores how well each candidate's evidence supports the query's structure, but it does not drop candidates:

1. For each `relationship` (`subject relation object [context]`) and each free-text `requirement`, a hypothesis string is built.
2. Every hypothesis is scored against every evidence chunk of every candidate in batched calls to the remote BGE cross-encoder (`/rerank` endpoint, shared with Phase 6's reranker server).
3. **Relationship support**: `SUPPORTED` if a deterministic word-proximity check finds the subject's and object's terms co-occurring within `max_word_distance` (40) words of each other in some chunk (`check_span_proximity`), **or** the best cross-encoder score across the candidate's chunks is ≥ `VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD` (0.35).
4. **Requirement support**: `SUPPORTED` if the best of (cross-encoder score, `0.5 ×` word-overlap ratio) across the candidate's chunks is ≥ `VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD` (0.35).
5. Per-patent `relationship_coverage` / `requirement_coverage` are the fraction of relationships/requirements marked supported (1.0 if the query has none of that kind, or is metadata-only). These ratios feed Phase 7's score; **no candidate is excluded here**, however low its coverage.

---

### Stage 11: BGE Cross-Encoder Reranking & Highlighting (`app/reranking/reranker.py` + `app/highlighting/`, Search Phase 6)
1. `build_rerank_query` deterministically builds one reranking query from `semantic_query` (or `original_query`), `relationships`, and `requirements` — no LLM call, no new embeddings.
2. Each evidence chunk is formatted (`Section: <name>\n\n<text>`) and truncated to a token budget (`RERANKER_MAX_CONTEXT_TOKENS` minus the query's tokens minus `RERANKER_TOKEN_SAFETY_MARGIN`), sliced at exact tokenizer offsets so truncation never splits a token.
3. All chunks across all verified candidates are scored against the reranking query in one set of batched HTTP calls (`RERANK_BATCH_SIZE` docs per request, up to `RERANK_CONCURRENT_REQUESTS` in flight) to the remote BGE reranker (`BAAI/bge-reranker-v2-m3`). A failed batch scores 0.0 rather than aborting the search.
4. **Query-match highlighting**: each chunk's sentences (split multilingually, short fragments like "FIG. 1" skipped) are scored in the **same** batched requests as the chunks, and sentences scoring above `HIGHLIGHT_SENTENCE_THRESHOLD` (strong above `HIGHLIGHT_SENTENCE_STRONG_THRESHOLD`) are marked. Separately, the query's own content words (stemmed, CJK-aware) are matched as literal term spans. Both are character-offset spans the UI highlights directly on the chunk text.
5. A patent's `best_reranker_score` is the **MAX** across its own chunks' scores (not an average); `avg_reranker_score` is also kept. Phase 5's verification results are carried through unchanged.

---

### Stage 12: Final Scoring & Result Selection (`app/scoring/scorer.py`, Search Phase 7)
Pure in-memory, deterministic — zero LLM/reranker/embedding/Qdrant calls. For each candidate, four 0–1 component scores are combined into one 0–10 final score:

```
final_score = 10 × ( 0.45 × relationship_score
                    + 0.25 × requirement_score
                    + 0.20 × reranker_score
                    + 0.10 × retrieval_score )
```

* `relationship_score` = Phase 5's `relationship_coverage`, forced to `0.0` if any relationship was explicitly `CONTRADICTED`.
* `requirement_score` = Phase 5's `requirement_coverage`.
* `reranker_score` = Phase 6's `best_reranker_score`, clamped to [0, 1].
* `retrieval_score` = Phase 2's `candidate_score` (retrieval similarity), clamped to [0, 1].
* The four weights (`FINAL_WEIGHT_*`) must sum to 1.0 (validated at `FinalScorer` construction).
* A patent qualifies only if `final_score >= FINAL_SCORE_THRESHOLD` (default 7.0); qualifying patents are returned sorted descending, with **no fixed top-K cap**.
* **Metadata-only queries** skip scoring entirely: there is no topic to verify or rank against, so every matching patent is returned unscored and unranked, in the order Phase 6 delivered them, rather than dressing up a meaningless number.

---

## 🛠️ Project Configuration & Tunables (`app/config.py`)

| Component | Setting | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Qdrant** | `QDRANT_HOST` / `PORT` | `localhost:6333` | Qdrant vector database connection |
| | `CHUNKS_COLLECTION_PREFIX` / `PATENTS_COLLECTION_PREFIX` | `"patent_chunks_"` / `"patents_metadata_"` | Naming convention search uses to discover collections |
| | `CHUNKS_COLLECTION_NAME` / `PATENTS_COLLECTION_NAME` | `"patent_chunks_512"` / `"patents_metadata_512"` | The collection pair ingestion writes to |
| **Embedding** | `EMBEDDING_REMOTE_BASE_URL` / `_MODEL` | — | Remote vLLM embedding server & `Qwen/Qwen3-Embedding-0.6B` |
| | `VECTOR_SIZE` | `1024` | Vector dimensionality |
| **Chunking** | `MAX_CHUNK_TOKENS` | `512` | Token capacity limit per chunk |
| **Ingestion** | `EMBED_BATCH_SIZE` / `EMBED_CONCURRENT_REQUESTS` | `128` / `5` | Texts per embedding HTTP request / parallel requests in flight |
| | `CHUNK_QUEUE_CAPACITY` | `1024` | Chunks buffered before the first embedding window is cut |
| | `BATCH_SIZE` / `INSERT_WORKERS` | `512` / `4` | Points per Qdrant insertion batch / parallel insert workers |
| | `INGEST_PROGRESS_FILE` | `"data/ingest_progress_512.log"` | Resume checkpoint log |
| **Validator** | `VALIDATOR_LOW_INFO_THRESHOLD` | `0.30` | Minimum ratio of alpha characters required |
| | `VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD` | `0.9` | Minimum unique-content ratio vs. previous chunk |
| **Query Understanding** | `QUERY_LLM_REMOTE_BASE_URL` / `_MODEL` | — | Remote OpenAI-compatible endpoint & model |
| | `QUERY_CACHE_SIZE` | `1024` | LRU cache size for parsed queries |
| **Phase 2 Retrieval** | `RETRIEVAL_TOP_K_PER_VIEW` | `500` | Chunks fetched per retrieval view before dedup/grouping |
| | `PATENT_CANDIDATE_TOP_K` | `300` | Candidate patents kept after Phase 2 grouping |
| **Phase 4 Evidence** | `EVIDENCE_GLOBAL_TOP_K_CHUNKS` | `1000` | Global (not per-patent) cap on the evidence vector search |
| | `EVIDENCE_NEIGHBOR_CHUNKS` | `1` | Adjacent chunk indices fetched on each side of a matched chunk |
| **Phase 5 Verification** | `VERIFICATION_MAX_CANDIDATES` | `None` | Caps candidates verified (and thus Phase 6/7); `None` = verify all |
| | `VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD` | `0.35` | Min cross-encoder score to mark a relationship SUPPORTED |
| | `VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD` | `0.35` | Min effective score to mark a requirement SUPPORTED |
| **Phase 6 Reranker** | `RERANKER_REMOTE_BASE_URL` / `_MODEL` | — | Remote BGE cross-encoder server & `BAAI/bge-reranker-v2-m3` |
| | `RERANK_BATCH_SIZE` / `RERANK_CONCURRENT_REQUESTS` | `128` / `6` | Docs per rerank HTTP request / requests in flight |
| | `RERANKER_MAX_CONTEXT_TOKENS` / `_TOKEN_SAFETY_MARGIN` | `4096` / `16` | Combined query+doc token budget per request |
| **Phase 7 Scoring** | `FINAL_SCORE_THRESHOLD` | `7.0` | Minimum 0–10 final score to appear in results |
| | `FINAL_WEIGHT_RELATIONSHIP/REQUIREMENT/RERANKER/RETRIEVAL` | `0.45 / 0.25 / 0.20 / 0.10` | Multi-signal scoring weights (must sum to 1.0) |
| **Highlighting** | `HIGHLIGHT_SENTENCE_THRESHOLD` / `_STRONG_THRESHOLD` | `0.02` / `0.25` | Cross-encoder sentence score cutoffs for highlighting |
| | `HIGHLIGHT_MIN_SENTENCE_CHARS` / `_WORDS` | `8` / `4` | Minimum sentence size scored for highlighting |
| **UI** | `PATENT_VIEW_URL_TEMPLATE` | — | External patent detail page URL template |

---

## 📁 Repository Directory Structure

```
rag_demo/
├── app/
│   ├── chunking/                       # Section detection & token-window chunking
│   │   ├── section_detector.py         # Known-heading-whitelist section boundary detection
│   │   ├── known_headings.py           # Whitelist of recognized patent section headings
│   │   ├── token_window_chunker.py     # Token-bounded window chunker with boundary adjustment
│   │   ├── chunk_validator.py          # Quality and duplicate filtering
│   │   └── token_counter.py            # HuggingFace token counter with LRU cache
│   ├── query_understanding/            # LLM-based natural-language query parsing
│   │   ├── engine.py                   # QueryUnderstandingEngine: query -> ParsedQuery (+ cache, JSON repair)
│   │   ├── cache.py                    # LRU query cache
│   │   ├── prompt.py                   # Query-understanding system prompt
│   │   ├── field_mapping.py            # Filterable metadata field allowlist
│   │   └── normalizer.py               # Country / organization name normalization
│   ├── retrieval/                      # Phases 2-4: candidate + evidence retrieval
│   │   ├── retriever.py                # CandidateRetriever: multi-view vector search (Phase 2)
│   │   ├── views.py                    # Deterministic original/semantic/structured view builder
│   │   ├── filter_builder.py           # Metadata field -> Qdrant filter / payload key resolution
│   │   ├── metadata_filter.py          # Strict-AND metadata constraint enforcement (Phase 3)
│   │   └── evidence_retriever.py       # Bounded evidence chunk + neighbor retrieval (Phase 4)
│   ├── verification/
│   │   └── verifier.py                 # RelationshipVerifier: cross-encoder + proximity coverage (Phase 5)
│   ├── reranking/
│   │   └── reranker.py                 # BGEReranker: cross-encoder reranking + highlighting (Phase 6)
│   ├── highlighting/
│   │   └── highlighter.py              # Sentence/term span highlighting used by the reranker
│   ├── scoring/
│   │   └── scorer.py                   # FinalScorer: weighted multi-signal scoring (Phase 7)
│   ├── models/                         # Pydantic dataclasses & schema definitions
│   │   ├── parsed_query.py             # ParsedQuery, SemanticRelationship, ConceptAttribute, MetadataFilter
│   │   ├── candidate.py                # CandidateChunk, CandidatePatent, (Filtered)CandidateRetrievalResult
│   │   ├── evidence.py                 # EvidenceChunk, PatentEvidence, EvidenceRetrievalResult
│   │   ├── verification.py             # RelationshipVerification, RequirementVerification, VerificationBatchResult
│   │   ├── reranking.py                # RerankedEvidenceChunk, RerankedPatentResult, ChunkHighlight
│   │   ├── scoring.py                  # FinalPatentResult, ScoreBreakdown, FinalSearchResult
│   │   ├── collection.py               # SearchCollection (chunks + patents collection pair)
│   │   ├── patent_chunk.py             # PatentChunk dataclass (ingestion)
│   │   └── patent_document.py          # Raw parsed PatentDocument dataclass (ingestion)
│   ├── api/
│   │   ├── main.py                     # FastAPI server: streams the 7-phase pipeline as NDJSON
│   │   ├── schemas.py                  # Request/response Pydantic models
│   │   ├── history.py                  # Search history endpoints (PostgreSQL)
│   │   ├── settings.py                 # UI appearance settings endpoints
│   │   └── memory.py                   # Process memory sampling for /api/compare
│   ├── db/                             # PostgreSQL search-history persistence (SQLAlchemy)
│   ├── scripts/
│   │   ├── show_indexed_patents.py     # Summary table generator for all Qdrant-indexed patents
│   │   └── build_index.py              # Resume/watch the Qdrant HNSW index build after ingest
│   ├── _tests_/                        # Component diagnostic scripts (no pytest required)
│   ├── config.py                       # Project configuration & hyperparameter tunables
│   ├── parser.py                       # Reads .txt and .json patent source files
│   ├── chunker.py                      # Orchestrates the chunking subsystem
│   ├── embedder.py                     # Remote vLLM embedding client
│   ├── qdrant_db.py                    # Qdrant client: collection discovery, search, metadata I/O
│   ├── ingest.py                       # Five-stage concurrent ingestion pipeline
│   └── semantic_search.py              # SearchPipeline: orchestrates Phases 1-7
├── frontend/                           # Vite + React + TypeScript UI (see frontend/README.md)
├── docker-compose.yml                  # Qdrant + PostgreSQL + Adminer + API (production-style)
├── docker-compose.override.yml         # Dev overrides: hot-reload API + Vite dev server
├── requirements.txt                    # Python dependencies
└── README.md                           # Comprehensive documentation
```

---

## 🚀 Quickstart & Usage Guide

### 1. Prerequisites & Installation
* **Python**: `3.10` or higher
* **Docker & Docker Compose**: Installed and running

Install python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start Qdrant Vector Database
Start the containerized Qdrant instance:
```bash
docker-compose up -d
```
Verify Qdrant is running at `http://localhost:6333/dashboard`.

### 3. Run Ingestion Pipeline
To ingest, chunk, embed, and index patents into Qdrant:
```bash
./venv/bin/python -m app.ingest
```
This creates the `patent_chunks_<name>` / `patents_metadata_<name>` collection pair named by `CHUNKS_COLLECTION_NAME` / `PATENTS_COLLECTION_NAME` on first run, and resumes from `INGEST_PROGRESS_FILE` if interrupted. To wipe and recreate both collections from scratch:
```bash
./venv/bin/python -c "from app.qdrant_db import QdrantDB; QdrantDB().reset_collections()"
```
If an ingest is interrupted before the final index build, finish or re-watch it with:
```bash
./venv/bin/python -m app.scripts.build_index
```

### 3.1 View All Indexed Patents Summary Table
To view a complete table of all indexed patents in Qdrant with chunk counts, section statistics, and token totals:
```bash
./venv/bin/python -m app.scripts.show_indexed_patents
```

For detailed section-by-section breakdown:
```bash
./venv/bin/python -m app.scripts.show_indexed_patents -v
```

### 4. Run the Web App (React + FastAPI)
The React frontend (`frontend/`) talks to a FastAPI server (`app/api/main.py`) that runs the
`SearchPipeline` and streams each phase's result as it completes.

**With Docker (next to Qdrant):** `docker compose up -d` starts everything in development
mode (`docker-compose.override.yml` is merged in automatically):

* UI with hot reload on **http://localhost:5173** (the `frontend` service, Vite dev server).
  UI pages on :8000 redirect there.
* API on http://localhost:8000. `app/` is mounted into the container and uvicorn restarts
  on every `.py` change.

No rebuild is needed for code changes. Rebuild `web` only after changing `requirements.txt`
(`docker compose up -d --build web`). For the production-style image, with the UI built into it
and served on :8000, skip the override: `docker compose -f docker-compose.yml up -d --build`.

Every search is saved to PostgreSQL (the `postgres` service, host port 5434) and can be
reviewed on the **History** page. Tables are created automatically on API startup. Outside
Docker the API uses `DATABASE_URL` (default `postgresql+psycopg://patent:patent@localhost:5434/patent_search`),
so run `docker compose up -d postgres` first when using `./run_api.sh`.

```bash
docker compose up -d                # qdrant + postgres + web + frontend
docker compose logs -f web          # follow API logs (shows reloads)
```

**Local development:**

```bash
# Terminal 1 — API on http://localhost:8000
./run_api.sh

# Terminal 2 — dev UI on http://localhost:5173 (proxies /api to :8000)
cd frontend && npm install && npm run dev
```

For a single-server setup, run `npm run build` in `frontend/`; the API then serves the built UI at
http://localhost:8000.

Frontend stack: Vite + React + TypeScript, Tailwind CSS, Redux Toolkit (search/UI state),
React Query (server calls), Zod (form + API response validation).

| Folder | Contents |
| --- | --- |
| `src/components/ui/` | Generic reusable UI kit (Button, Card, Drawer, DataTable, Tabs, …) |
| `src/components/patent/` | Patent-specific reusable pieces (heading, evidence chunk, verification list) |
| `src/features/` | Screens: search, compare, pipeline, results, history, settings |
| `src/schemas/` | Zod schemas — mirror the Pydantic models in `app/models/`; types come from `z.infer` |
| `src/store/` | Redux slices and selectors |
| `src/hooks/` | React Query hooks, including the streaming search hook |

### 4.1 Comparing Collections
`POST /api/compare` parses a query once and runs Phases 2–7 against several collections in turn, streaming per-collection phase events plus process-memory usage, so collections (e.g. different `MAX_CHUNK_TOKENS` sizes) can be compared head-to-head. Exposed in the UI's **Compare** page; compare runs are not saved to search history.

### 5. Running Component Verification Scripts
Diagnostic scripts live under `app/_tests_/` as plain Python scripts (no pytest required):
```bash
./venv/bin/python -m app._tests_.test_chunker
./venv/bin/python -m app._tests_.test_token_window_chunker
./venv/bin/python -m app._tests_.test_parser
./venv/bin/python -m app._tests_.test_embedding
./venv/bin/python -m app._tests_.test_connection
./venv/bin/python -m app._tests_.test_qdrant
./venv/bin/python -m app._tests_.test_batch_insert
./venv/bin/python -m app._tests_.test_reset_collection
```

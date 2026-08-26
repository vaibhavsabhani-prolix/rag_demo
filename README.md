# Patent Semantic Search System

A production-ready Patent Retrieval & Semantic Search system built with **Python**, **Qdrant Vector DB**, the **Qwen3 Embedding Model**, an **LLM-based Query Understanding layer**, **metadata filtering**, a **cross-encoder reranker**, and an **Answer Evidence Extraction & Highlighting engine**.

The pipeline handles end-to-end processing of complex technical patent documents: from raw document parsing and multi-stage semantic chunking, through validation, vector embedding, and batch indexing — to natural-language query understanding, answer-target-preserving semantic retrieval for questions, patent-grouped vector search, post-retrieval metadata filtering, cross-encoder reranking, evidence selection, and patent-level result aggregation. A Streamlit UI and CLI tools are included for search and index inspection.

---

## 📐 System Architecture & Flow

```mermaid
flowchart TD
    subgraph Ingestion_Pipeline ["1. Ingestion & Indexing Pipeline"]
        A["Raw Patent Files (.txt + .json)"] --> B["Patent Parser (app/parser.py)"]
        B --> C["Patent Document Object"]
        C --> D["Section Detector (app/chunking/section_detector.py)"]
        D --> E["Semantic Unit Splitter (app/chunking/semantic_unit_splitter.py)"]
        E --> F["Token-Aware Chunk Builder (app/chunking/chunk_builder.py)"]
        F --> G["Chunk Validator (app/chunking/chunk_validator.py)"]
        G --> H["Embedding Engine (Qwen3-Embedding-0.6B)"]
        H --> I1["patent_chunks collection (vectors + chunk payload)"]
        C --> I2["patents collection (metadata only, no vectors)"]
    end

    subgraph Search_Pipeline ["2. Query Understanding & Search Pipeline"]
        J["User Search Query"] --> K["Query Understanding LLM (app/query_understanding/)"]
        K --> K1["ParsedQuery: semantic_query + metadata_filters +\nis_question + question_intent + dynamic requirements"]
        
        K1 -->|"is_metadata_only: filters, no topic"| MO1["Filter whole patents collection\n(QdrantDB.filter_patent_ids)"]
        MO1 --> MO2["Fetch every chunk of matching patents\n(QdrantDB.get_chunks_for_patent_ids) - unbounded, no rerank"]
        MO2 --> P
        
        K1 -->|"Topic Query (is_question=false)"| L1["semantic_query = core topic\n(e.g., 'material property determination')"]
        K1 -->|"Question Query (is_question=true)"| L2["semantic_query = answer-target-preserving retrieval query\n(e.g., 'properties determined based on identified material')"]
        
        L1 --> L["Embed semantic_query ONLY (app/embedder.py)"]
        L2 --> L
        
        L --> M["Qdrant group-by-patent_id search on patent_chunks\n(top PATENT_CANDIDATE_TOP_K patents - identifies WHICH\npatents are candidates, not which of their chunks matter)"]
        M --> M1["Dedup to 1 best chunk/patent for display only\n(_dedupe_top_chunk_per_patent -> qdrant_results)"]
        M --> N{"metadata_filters present?"}
        N -->|"yes"| N1["FilterEngine.matches() per candidate patent_id\n(metadata lookup only, before any chunk text is fetched)"]
        N -->|"no"| F1
        N1 --> F1["Fetch EVERY indexed chunk of each surviving\ncandidate patent (QdrantDB.get_chunks_for_patent_ids -\nunbounded, not just the handful vector search surfaced)"]
        F1 --> O["Cross-Encoder Reranker (remote or local, app/reranker.py)\nScores EVERY one of those chunks individually;\na patent's score = MAX across its own chunks -\nno hand-tuned weights, no combined-text blob"]
        O --> O1["Exclusion hard-filter: drop a patent only if its OWN\nwinning (highest-scoring) chunk names an excluded term"]
        O1 --> O2["Keep only patents scoring >= PATENT_RELEVANCE_THRESHOLD\n(0-10 scale; a single hard cutoff)"]

        O2 --> ES{"is_question?"}

        ES -->|"yes"| ES1["Evidence Selector & Answer Extractor (app/evidence_selector.py)\nExtracts answer, verbatim evidence, confidence, and character spans"]
        ES -->|"no"| P["Patent Aggregation (app/semantic_search.py)"]
        ES1 --> P
        
        P --> Q["PatentSearchResult list, top FINAL_TOP_K\n(question queries: a patent with a genuine\nanswer leads, otherwise sorted by relevance score)"]
        Q --> R["Streamlit UI (app/ui/search_app.py) / CLI"]
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

### Stage 2: Section Detection & Semantic Chunking (`app/chunker.py` + `app/chunking/*`)
Patents require strict structural isolation — chunks must never mix contents across document sections.

* **Step 2.1 - Section Detector (`section_detector.py`)**:
  * Scans document text line-by-line using heading heuristics (all-caps line, colon-terminated titles, short line length bounds).
  * Identifies standard patent sections (e.g., `ABSTRACT`, `BACKGROUND OF THE INVENTION`, `SUMMARY`, `DETAILED DESCRIPTION`, `CLAIMS`).
  * Yields isolated `Section` objects. Each section is processed as an independent stream.

* **Step 2.2 - Hierarchical Semantic Unit Splitter (`semantic_unit_splitter.py`)**:
  * Splits section content hierarchically: **Paragraphs → Sentences → Word Fragments**.
  * Measures precise token lengths using `TokenCounter` backed by the HuggingFace `Qwen3-Embedding` tokenizer with an LRU cache (`TOKEN_COUNT_CACHE_SIZE = 4096`).

* **Step 2.3 - Token-Aware Chunk Builder (`chunk_builder.py`)**:
  * Merges semantic units greedily until reaching `MAX_CHUNK_TOKENS` (default: 512 tokens).
  * Section boundaries are strictly enforced — no chunk spans multiple sections.

---

### Stage 3: Chunk Quality Validation & Filtering (`app/chunking/chunk_validator.py`)
To prevent indexing low-quality vector noise into Qdrant, every chunk passes through rigorous validation rules:

1. **Whitespace Normalization**: Collapses redundant tabs, double spaces, and newline padding.
2. **Heading-Only Rejection**: Filters out isolated headings without body content (e.g., `DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS`).
3. **Low-Information Filtering**: Calculates the ratio of alphabetic characters to total characters. Rejects chunks falling below `VALIDATOR_LOW_INFO_THRESHOLD` (0.30) to eliminate table artifacts, binary noise, and separator lines.
4. **Degenerate Overlap Loop Prevention**: Tracks unique token ratios against previous chunks to reject near-duplicate chunks.

---

### Stage 4: Vector Embedding Generation (`app/embedder.py`)
* **Model**: `Qwen/Qwen3-Embedding-0.6B` via `sentence-transformers`.
* **Output Dimension**: 1024-dimensional dense vectors (`VECTOR_SIZE = 1024`).
* **Normalization**: L2 normalization (`normalize_embeddings=True`) applied to all chunk and query vectors for accurate cosine similarity calculation.
* **Process**: Each validated `PatentChunk` is passed to `Embedder.embed()`, populating its `.vector` attribute.

---

### Stage 5: Vector DB Indexing & Storage (`app/qdrant_db.py` & `app/ingest.py`)
* **Vector Store**: **Qdrant** running via Docker on port `6333`.
* **Two Collections**:
  * `patent_chunks` — one point per chunk, with its embedding vector and full chunk payload. Searched.
  * `patents` — one point per patent, metadata only, no vectors. Looked up by `patent_id` for display and metadata filtering; never searched by vector.
* **Distance Metric**: Cosine Similarity.
* **Batch Ingestion**:
  * `app/ingest.py` calls `QdrantDB.create_collections()` on startup (idempotent — no-ops if they already exist), so a fresh Qdrant instance gets both collections created automatically on first run; no manual setup step is required.
  * It then orchestrates ingestion of patent files from `PATENT_DIRECTORY` (`app/config.py`, currently `"television"`).
  * Accumulates embedded chunks in batches of `BATCH_SIZE = 100` and upserts via `QdrantDB.insert_batch()`.
  * Upserts each patent's metadata once via `QdrantDB.upsert_patent_metadata()`.
* **Chunk Payload Attributes**:
  * `patent_id`, `section`, `text`, `chunk_id`, `section_chunk_index`, `document_chunk_index`, `total_chunks`, `token_count`, `word_count`.

---

### Stage 6: Query Understanding & Question Processing (`app/query_understanding/`)
Before anything is embedded, the raw natural-language query is processed by the Query Understanding LLM (remote endpoint with local fallback). It classifies the query type and extracts structured representations into a `ParsedQuery`:

1. **Query Intent Classification (`is_question`)**:
   * **Topic Search (`is_question = false`)**: User wants to find patents about a topic or technology (e.g., *"material-aware 3D scanning"*, *"methods for determining material properties"*).
   * **Question / Answer-Seeking Query (`is_question = true`)**: User seeks specific factual information from patent text (e.g., *"What types of image capture devices can be used in the system?"*, *"How is material determined based on detected features?"*).

2. **Adaptive Semantic Query Generation (`semantic_query`)**:
   * **Rule 4A (Topic Search)**: Strips recognized metadata filter phrases and outputs the clean invention/technology topic.
   * **Rule 4B (Question Search)**: Generates an answer-target-preserving retrieval query:
     $$\text{semantic\_query} = \text{requested\_answer\_target} + \text{target\_entity} + \text{important\_relationship}$$
     * Preserves the exact answer target (*types of devices*, *properties*, *methods*, *reasons*, *components*, *differences*) and qualifying context (*used in the system*, *based on detected features*).
     * Prevents over-summarization (does **not** reduce *"What types of image capture devices can be used in the system?"* to *"image capture devices"* or *"image capture system"*).
     * Avoids over-expansion and hallucinated keyword dumps.

3. **Metadata Filters (`metadata_filters`)**:
   * Extracts structured `(field, operator, value)` constraints mapped strictly to the allowlist (e.g. `CAN_EN`, `AAPS`, `PY`, `PRC`). Filters are preserved separately from `semantic_query` even in question queries.

4. **Dynamic Requirements Structure & Question Intent**:
   * For topic searches: extracts `concepts`, `goals`, `constraints`, `optimization`, `exclusions`, and `relationships`. Only `exclusions` feeds the reranker (its hard-filter, see Stage 9) - the rest are used to highlight matched query terms in the displayed result text (`app/evidence_selector.py`).
   * For questions: populates `question_intent` with `target`, `expected_answer_type`, and `answer_criteria`.

---

### Stage 6B: Metadata-Only Fast Path (`SemanticSearch._search_by_metadata_only`)
Taken whenever `parsed.is_metadata_only` is `True` (e.g. *"applications filed in 2008 by Wyeth"*):

1. `QdrantDB.filter_patent_ids()` scrolls the entire `patents` collection and returns every `patent_id` matching all `metadata_filters`.
2. `QdrantDB.get_chunks_for_patent_ids()` fetches every chunk belonging to those patents from `patent_chunks` without vector search or reranking.
3. Chunks slot directly into aggregation with a placeholder score of `10.0` (max of the reranker's 0-10 scale) - a confirmed metadata match, not an actual relevance judgment, since there's no topic text to score against.

---

### Stage 7: Semantic Vector Search (`app/semantic_search.py` + `app/qdrant_db.py`)
1. `Embedder.embed_query(parsed.semantic_query)` embeds the retrieval-optimized `semantic_query`.
2. `QdrantDB.search()` executes a **group-by-`patent_id`** vector search against `patent_chunks`, retrieving the top `PATENT_CANDIDATE_TOP_K` (default: 50) candidate patents, each contributing up to `CANDIDATE_CHUNKS_PER_PATENT` (default: 3) chunks. This step decides **which patents** are candidates (a cheap embedding-similarity signal) - it is NOT the chunk set reranking sees (see Stage 8b).
3. Two internal views are created:
   * `candidate_chunks`: used only to derive the candidate `patent_id` list and the display view below.
   * `qdrant_results`: a deduplicated display-only view (top chunk per patent) for candidate inspection - shown in the UI as "Qdrant Vector Search Candidates". It does not reflect the fuller chunk set reranking actually checks.

---

### Stage 8a: Metadata Filtering, on Candidate Patent IDs (`app/filter_engine.py`)
Runs BEFORE any chunk text is fetched. If `metadata_filters` exist:
* Metadata for the candidate `patent_id`s from Stage 7 is batch-fetched from the `patents` collection.
* `FilterEngine.matches()` checks every metadata constraint against each candidate patent.
* Non-matching patent_ids are dropped - so a patent's full chunk set (Stage 8b) is only ever pulled for patents that already pass this.

---

### Stage 8b: Full Chunk Retrieval (`QdrantDB.get_chunks_for_patent_ids`)
For each surviving candidate patent_id, fetches **every** chunk it has in `patent_chunks` - unbounded, not just the `CANDIDATE_CHUNKS_PER_PATENT` chunks the initial vector search happened to surface. Patents in this corpus range from 1 to over 2,000 chunks, so this is the step that lets reranking judge a patent by its strongest evidence out of everything it discloses, not a similarity-biased subset.

---

### Stage 9: Cross-Encoder Reranking (`app/reranker.py`)
Scores every candidate **chunk individually**, then assigns each patent the MAX of its own chunks' scores:
1. Every chunk fetched in Stage 8b is scored against `semantic_query` in **one batched cross-encoder call** (remote reranking server with local `CrossEncoder` fallback), producing a **0-10 relevance score** per chunk.
2. A patent's score is the **MAX** across all of its own chunks' scores - it's judged by its single strongest disclosed passage, not an average, not a top-K cut, and not a combined-text blob (a patent can have thousands of chunks, so combining them would exceed any usable model input length). The chunk that earned that max is the patent's best-evidence chunk, stashed on `chunk.payload["_chunk_relevance"]`.
3. **Exclusion hard-filter**: if the query carries exclusion terms (e.g. *"without indium tin oxide"*), a patent is dropped only if its **own winning chunk** - the specific evidence that earned it its score - literally names an excluded term. Only that one chunk is checked, not the patent's whole chunk set: Query Understanding's exclusion extraction is itself an LLM call and can occasionally infer a term the user never asked to exclude, and a long patent will often mention an ordinary, unrelated term like that somewhere irrelevant - checking every chunk would let a hallucinated exclusion wrongly sink an otherwise correct match.
4. `SemanticSearch.search_detailed()` then keeps only patents scoring at or above `PATENT_RELEVANCE_THRESHOLD` (default `7.0`) — the only threshold in the whole scoring path; there is no hand-tuned weighted blend.

---

### Stage 10: Answer Evidence Selection & Span Highlighting (`app/evidence_selector.py`)
For question queries (`parsed.is_question = True`), candidate chunks pass through the `EvidenceSelector`:

1. **Answer & Evidence Extraction**:
   * Analyzes top reranked chunks against `parsed.question_intent`.
   * Extracts a concise **answer summary** and verbatim **supporting evidence sentences**.
   * Employs remote LLM extraction with an automated deterministic fallback algorithm based on target directness and core term coverage.
2. **Exact Character Span Resolution**:
   * `locate_span_in_text()` calculates precise character offsets `[start_char, end_char]` inside the original chunk text.
3. **HTML Highlighting**:
   * `highlight_spans()` wraps matched evidence in `<mark>` tags for UI presentation.
4. Attaches `answer`, `answer_evidence`, `answer_span`, `answer_score`, and `highlighted_text` to the chunk payload.

---

### Stage 11: Patent-Level Aggregation (`app/semantic_search.py`)
* `SemanticSearch._aggregate_by_patent()` groups reranked chunks by `patent_id`.
* **Patent Score**: the MAX-of-its-chunks relevance score from Stage 9 (identical across a patent's own chunks, since it's the same max); `best_chunk` is the specific chunk whose own score (`RankedChunk.chunk_relevance_score`) produced that max - the same model that decided relevance also decides which passage to display, not a separate heuristic.
* **Sorting**: question queries put a patent with a genuine extracted answer first, ahead of any patent without one (regardless of relevance score); topic queries sort by relevance score alone.
* Preserves question answers and highlighted evidence in `PatentSearchResult`.
* Results are truncated to `FINAL_TOP_K` (default: 10) patents.

---

## 🛠️ Project Configuration & Tunables (`app/config.py`)

All system thresholds are centrally managed in `app/config.py`:

| Component | Setting | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Qdrant** | `QDRANT_HOST` / `PORT` | `localhost:6333` | Qdrant vector database connection |
| | `CHUNKS_COLLECTION_NAME` | `"patent_chunks"` | Searchable chunk collection (vectors + payload) |
| | `PATENTS_COLLECTION_NAME` | `"patents"` | Patent metadata collection (no vectors) |
| **Embedding** | `EMBEDDING_MODEL` | `"Qwen/Qwen3-Embedding-0.6B"` | SentenceTransformer embedding model |
| | `VECTOR_SIZE` | `1024` | Vector dimensionality |
| **Chunking** | `MAX_CHUNK_TOKENS` | `512` | Token capacity limit per chunk |
| | `MIN_CHUNK_TOKENS` / `MIN_CHUNK_WORDS` | `20` / `8` | Minimum size for a valid chunk |
| | `BATCH_SIZE` | `100` | Points per Qdrant upload batch |
| **Validator** | `VALIDATOR_LOW_INFO_THRESHOLD` | `0.30` | Minimum ratio of alpha characters required |
| | `VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD` | `0.9` | Minimum unique-content ratio vs. previous chunk |
| **Query Understanding** | `QUERY_LLM_REMOTE_BASE_URL` / `_MODEL` | — | Remote OpenAI-compatible endpoint & model |
| **Reranker** | `RERANKER_REMOTE_BASE_URL` / `_MODEL` | — | Remote reranking server URL & model |
| | `PATENT_CANDIDATE_TOP_K` | `50` | Distinct candidate patents from vector search - each gets EVERY one of its own chunks checked by reranking, so this is the main lever on reranking latency vs. candidate breadth |
| | `CANDIDATE_CHUNKS_PER_PATENT` | `3` | Chunks per candidate patent from the INITIAL vector-search step only (identifying candidates + the display view) - reranking itself checks a patent's complete chunk set, not this |
| | `PATENT_RELEVANCE_THRESHOLD` | `7.0` | 0-10 relevance score (the MAX across a patent's own chunks) a patent must meet to be kept as a match - the only reranking threshold |
| | `FINAL_TOP_K` | `10` | Final reranked patents returned |
| **UI** | `PATENT_VIEW_URL_TEMPLATE` | — | External patent detail page URL template |

---

## 📁 Repository Directory Structure

```
rag_demo/
├── app/
│   ├── chunking/                       # Token-aware chunking subsystem
│   │   ├── chunk_builder.py            # Greedily constructs token-bounded chunks
│   │   ├── chunk_validator.py          # Quality and duplicate filtering
│   │   ├── section_detector.py         # Heading & patent section boundary detection
│   │   ├── semantic_unit_splitter.py   # Paragraph/sentence semantic unit splitter
│   │   └── token_counter.py            # HuggingFace token counter with LRU cache
│   ├── query_understanding/            # LLM-based natural-language query parsing
│   │   ├── parser.py                   # QueryUnderstanding: query -> ParsedQuery
│   │   ├── models.py                   # ParsedQuery, MetadataFilter, Concept, QuestionIntent, etc.
│   │   ├── prompt.py                   # Query-understanding prompt template (Rules 4A/4B, 7)
│   │   ├── field_mapping.py            # Filterable metadata field allowlist
│   │   ├── metadata_field_codes.py     # Field code reference used in prompts
│   │   └── normalizer.py               # Country / organization name normalization
│   ├── models/                         # Dataclasses & schema definitions
│   │   ├── patent_chunk.py             # PatentChunk dataclass
│   │   ├── patent_document.py          # Raw parsed PatentDocument dataclass
│   │   ├── patent_search_result.py     # PatentSearchResult, RankedChunk, AnswerEvidence
│   │   └── search_result.py            # Shared search result helpers
│   ├── ui/
│   │   └── search_app.py               # Streamlit search UI with answer badges & highlight rendering
│   ├── scripts/
│   │   └── show_indexed_patents.py     # Summary table generator for all Qdrant-indexed patents
│   ├── _tests_/                        # Component & end-to-end test/diagnostic scripts
│   │   ├── test_semantic_search.py     # End-to-end search pipeline diagnostics
│   │   ├── test_reranker.py            # Whole-patent reranker & aggregation tests
│   │   ├── test_query_understanding.py # Query Understanding & field parsing tests
│   │   └── test_*.py                   # Per-component tests (chunker, qdrant, embedding, etc.)
│   ├── config.py                       # Project configuration & hyperparameter tunables
│   ├── parser.py                       # Reads .txt and .json patent source files
│   ├── chunker.py                      # Orchestrates the chunking subsystem
│   ├── embedder.py                     # HuggingFace Qwen embedding wrapper
│   ├── qdrant_db.py                    # Qdrant client: both collections, search, metadata I/O
│   ├── ingest.py                       # Ingestion pipeline script
│   ├── filter_engine.py                # Post-retrieval metadata filter evaluation
│   ├── reranker.py                     # Remote/local cross-encoder whole-patent reranking
│   ├── evidence_selector.py            # Answer evidence extraction & exact character span highlighting
│   └── semantic_search.py              # Patent-level semantic search coordinator
├── docker-compose.yml                  # Docker setup for Qdrant Vector DB
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
This automatically creates the `patent_chunks` and `patents` collections on first run. To wipe and recreate both collections from scratch:
```bash
./venv/bin/python -c "from app.qdrant_db import QdrantDB; QdrantDB().reset_collections()"
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

### 4. Run Semantic Search Diagnostics
Run the end-to-end search pipeline diagnostics tool:
```bash
./venv/bin/python -m app._tests_.test_semantic_search "Microdrilling"
```

### 4.1 Run the Search UI
Launch the interactive Streamlit search application:
```bash
./venv/bin/streamlit run app/ui/search_app.py
```
* Supports topic searches, metadata-filtered queries, and question queries.
* For questions, renders extracted answer summary badges and visual `<mark>` evidence highlighting.

### 5. Running Component Verification Tests
Component tests live under `app/_tests_/` as plain Python scripts (no pytest required):
* Test query understanding parser and prompt:
  ```bash
  ./venv/bin/python -m app._tests_.test_query_understanding
  ```
* Test the whole-patent reranker and patent aggregation:
  ```bash
  ./venv/bin/python -m app._tests_.test_reranker
  ```
* Test chunking, parsing, and embedding:
  ```bash
  ./venv/bin/python -m app._tests_.test_chunker
  ./venv/bin/python -m app._tests_.test_parser
  ./venv/bin/python -m app._tests_.test_embedding
  ```
* Test Qdrant connection and operations:
  ```bash
  ./venv/bin/python -m app._tests_.test_connection
  ./venv/bin/python -m app._tests_.test_qdrant
  ./venv/bin/python -m app._tests_.test_batch_insert
  ./venv/bin/python -m app._tests_.test_reset_collection
  ```


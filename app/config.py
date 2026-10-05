import os

# Overridable so the Docker web service can reach Qdrant by its compose service name.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_TIMEOUT = 120.0

# PostgreSQL for search history (the `postgres` service in docker-compose.yml).
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://patent:patent@localhost:5434/patent_search"
)

PATENT_DIRECTORY = "patents/bluetooth"

# Search discovers collections in Qdrant by name: every "patent_chunks_<name>"
# that has a matching "patents_metadata_<name>" can be searched as "<name>",
# and the user picks one in the UI. Name new collections this way.
CHUNKS_COLLECTION_PREFIX = "patent_chunks_"
PATENTS_COLLECTION_PREFIX = "patents_metadata_"

# The collection pair that ingestion writes to. Search preselects it when it exists.
CHUNKS_COLLECTION_NAME = "patent_chunks_512"
PATENTS_COLLECTION_NAME = "patents_metadata_512"

# Remote embedding server
EMBEDDING_REMOTE_BASE_URL = "http://192.168.2.213:8002/v1"
EMBEDDING_REMOTE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REMOTE_API_KEY = "EMPTY"
EMBEDDING_REQUEST_TIMEOUT = 3000.0
VECTOR_SIZE = 1024

QUERY_LLM_REMOTE_BASE_URL = "http://192.168.2.213:8000/v1"
# QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.8-27B-NVFP4"
QUERY_LLM_REMOTE_API_KEY = "EMPTY"
QUERY_LLM_REQUEST_TIMEOUT = 180.0
QUERY_LLM_TEMPERATURE = 0.0
QUERY_LLM_MAX_TOKENS = 1400
QUERY_CACHE_SIZE = 1024

RERANKER_REMOTE_BASE_URL = "http://192.168.2.213:8001"
RERANKER_REMOTE_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REMOTE_API_KEY = "EMPTY"
RERANKER_REQUEST_TIMEOUT = 360.0
RERANK_BATCH_SIZE = 128
RERANK_CONCURRENT_REQUESTS = 6
RERANKER_MAX_CONTEXT_TOKENS = 4096
RERANKER_TOKEN_SAFETY_MARGIN = 16

MAX_CHUNK_TOKENS = 512

EMBED_TOKEN_SAFETY_MARGIN = 8
BATCH_SIZE = 512
EMBED_BATCH_SIZE = 128
# The embedding GPU saturates at ~4 in-flight requests (measured:
# 94 chunks/s at 1, 114 at 4, still 114 at 16) - more only queues
# server-side.
EMBED_CONCURRENT_REQUESTS = 5
INSERT_REPORT_EVERY = 100
INGEST_PREFETCH = 512
CHUNK_QUEUE_CAPACITY = 1024
# Embedding caps the pipeline at ~114 points/s, so a few insert
# workers are plenty to keep up.
INSERT_WORKERS = 4

# Qdrant builds the HNSW search index for each segment as it fills up and
# rebuilds it whenever segments merge (collection status YELLOW), competing
# with a big ingest for CPU and disk. With this on, ingest_directory() pauses
# indexing of the chunks collection and builds the index once at the end.
# Until then, searches on that collection fall back to a slower full scan.
# Set INGEST_PAUSE_INDEXING=0 to index while ingesting instead.
INGEST_PAUSE_INDEXING = os.getenv("INGEST_PAUSE_INDEXING", "1") != "0"
# Qdrant's default: a segment holding more vector data than this (KB) gets an HNSW index.
QDRANT_INDEXING_THRESHOLD_KB = 20000

# Append-only log of patent filenames fully committed to Qdrant
# (metadata + every chunk). ingest_directory() reads it on startup to
# skip already-completed patents, so stopping and re-running the same
# command resumes instead of reprocessing from the start.
INGEST_PROGRESS_FILE = "data/ingest_progress_512.log"

MAX_HEADING_LENGTH = 100
MAX_HEADING_WORDS = 12
ALLCAPS_MIN_ALPHA = 3

VALIDATOR_NORMALIZE_WHITESPACE = True
VALIDATOR_REJECT_HEADING_ONLY = True
VALIDATOR_REJECT_LOW_INFO = True
VALIDATOR_LOW_INFO_THRESHOLD = 0.3
VALIDATOR_REJECT_DEGENERATE_OVERLAP = True
VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD = 0.9
VALIDATOR_HEADING_ONLY_MAX_WORDS = 3

DEBUG_CHUNKS = False
DEBUG_CHUNKS_DIR = "debug_chunks"

TOKEN_COUNT_CACHE_SIZE = 4096

PATENT_VIEW_URL_TEMPLATE = "https://www.qubeip.com/en/patent-view/{patent_id}"

# Phase 2 Retrieval configuration
RETRIEVAL_TOP_K_CHUNKS = 300

# Phase 4 Relationship Verification configuration
# Minimum cross-encoder relevance score required to mark a relationship/requirement as
# SUPPORTED. Was 0.05, which is far too permissive for a BGE cross-encoder — near-zero
# scores would still pass, letting patents that only share generic terms (e.g. "method",
# "manufacturing") with the query get marked as satisfying a specific requirement/relationship
# (e.g. "produces water") even though the key subject term never appears in the evidence.
# Raise this if unrelated patents are still slipping into results; lower it if genuinely
# relevant patents are being excluded.
VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD = 0.35
VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD = 0.35
# Minimum relationship_coverage / requirement_coverage a candidate must reach to
# survive Phase 4. A candidate below either cutoff is eliminated here and never
# reaches Phase 5/6, instead of just being scored lower.
VERIFICATION_RELATIONSHIP_COVERAGE_THRESHOLD = 0.5
VERIFICATION_REQUIREMENT_COVERAGE_THRESHOLD = 0.5

# Phase 6 Final Patent Scoring & Result Selection configuration
# Minimum final patent score required for a patent to appear in final results (0.0 to 10.0 scale).
# All qualifying patents are returned (no fixed top-K cap) — change this value to raise/lower the bar.
FINAL_SCORE_THRESHOLD = 7.0

# Multi-signal scoring weights (must sum to 1.0)
FINAL_WEIGHT_RELATIONSHIP = 0.45
FINAL_WEIGHT_REQUIREMENT = 0.25
FINAL_WEIGHT_RERANKER = 0.20
FINAL_WEIGHT_RETRIEVAL = 0.10

# Query match highlighting (computed in Phase 5 reranking)
# BGE cross-encoder score of each chunk sentence against the reranking query.
# Scores are sigmoid-shaped. Measured on a real "television" search (2087
# sentences): sentences mentioning TV had deciles 0.006-0.13, all others
# 0.0-0.011. 0.02 marked 173/278 TV sentences; 0.05 only 79.
HIGHLIGHT_SENTENCE_THRESHOLD = 0.02
HIGHLIGHT_SENTENCE_STRONG_THRESHOLD = 0.25
# Shorter fragments ("FIG. 1", claim numbers) are not scored.
HIGHLIGHT_MIN_SENTENCE_CHARS = 8
# Space-separated languages only: headings like "Electronic equipment" score
# very high on the cross-encoder without saying anything about the query.
HIGHLIGHT_MIN_SENTENCE_WORDS = 4



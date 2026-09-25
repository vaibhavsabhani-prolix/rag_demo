QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120.0

PATENT_DIRECTORY = "patents/2"

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
EMBED_BATCH_SIZE = 64
# The embedding GPU saturates at ~4 in-flight requests (measured:
# 94 chunks/s at 1, 114 at 4, still 114 at 16) - more only queues
# server-side.
EMBED_CONCURRENT_REQUESTS = 4
INSERT_REPORT_EVERY = 100
INGEST_PREFETCH = 512
CHUNK_QUEUE_CAPACITY = 1024
# Embedding caps the pipeline at ~114 points/s, so a few insert
# workers are plenty to keep up.
INSERT_WORKERS = 4

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
RETRIEVAL_TOP_K_PER_VIEW = 500  
PATENT_CANDIDATE_TOP_K = 300

# Phase 4 Bounded Evidence Retrieval configuration
# No per-patent chunk cap by design - every matched chunk and its neighbors
# are kept; EVIDENCE_GLOBAL_TOP_K_CHUNKS is the only ceiling, applied across
# the whole candidate batch during the Qdrant vector fetch.
EVIDENCE_NEIGHBOR_CHUNKS = 1
EVIDENCE_GLOBAL_TOP_K_CHUNKS = 1000

# Phase 5 Relationship Verification configuration
VERIFICATION_MAX_CANDIDATES = 25
# Minimum cross-encoder relevance score required to mark a relationship/requirement as
# SUPPORTED. Was 0.05, which is far too permissive for a BGE cross-encoder — near-zero
# scores would still pass, letting patents that only share generic terms (e.g. "method",
# "manufacturing") with the query get marked as satisfying a specific requirement/relationship
# (e.g. "produces water") even though the key subject term never appears in the evidence.
# Raise this if unrelated patents are still slipping into results; lower it if genuinely
# relevant patents are being excluded.
VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD = 0.35
VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD = 0.35

# Phase 7 Final Patent Scoring & Result Selection configuration
# Minimum final patent score required for a patent to appear in final results (0.0 to 10.0 scale).
# All qualifying patents are returned (no fixed top-K cap) — change this value to raise/lower the bar.
FINAL_SCORE_THRESHOLD = 7.0

# Multi-signal scoring weights (must sum to 1.0)
FINAL_WEIGHT_RELATIONSHIP = 0.45
FINAL_WEIGHT_REQUIREMENT = 0.25
FINAL_WEIGHT_RERANKER = 0.20
FINAL_WEIGHT_RETRIEVAL = 0.10



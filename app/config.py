"""
Project Configuration

All thresholds and tunables live here.
No magic numbers in component code.
"""

# ==========================
# Qdrant Configuration
# ==========================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Searchable chunk data (text, vector, lightweight fields). One point per chunk.
CHUNKS_COLLECTION_NAME = "patent_chunks"

# Patent metadata, stored once per patent. No vectors. One point per patent.
PATENTS_COLLECTION_NAME = "patents"

# ==========================
# Embedding Model
# ==========================

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Default output dimension of the model
VECTOR_SIZE = 1024

# ==========================
# Chunking Configuration
# ==========================

# Maximum tokens per chunk (measured by the embedding model tokenizer)
MAX_CHUNK_TOKENS = 512

# Minimum tokens for a chunk to be considered valid
MIN_CHUNK_TOKENS = 20

# Minimum words for a chunk to be considered valid
MIN_CHUNK_WORDS = 8

# Number of chunks to upload to Qdrant in one request
BATCH_SIZE = 100

# ==========================
# Section Detector
# ==========================

# Maximum character length for a line to be considered a heading
MAX_HEADING_LENGTH = 100

# Maximum word count for a line to be considered a heading
MAX_HEADING_WORDS = 12

# Minimum alphabetic characters for ALL-CAPS heading detection
ALLCAPS_MIN_ALPHA = 3

# ==========================
# Chunk Validator
# ==========================

# Normalize whitespace (collapse multiple spaces/newlines) before validation
VALIDATOR_NORMALIZE_WHITESPACE = True

# Reject chunks that contain only a heading-like line (short, ALL-CAPS, or colon-terminated)
VALIDATOR_REJECT_HEADING_ONLY = True

# Reject chunks with very low alphabetic content (tables, separators, numbering noise)
VALIDATOR_REJECT_LOW_INFO = True

# Ratio of alphabetic characters to total characters below which a chunk is "low info"
VALIDATOR_LOW_INFO_THRESHOLD = 0.3

# Reject chunks that are nearly identical to the previous chunk (degenerate overlap loops)
VALIDATOR_REJECT_DEGENERATE_OVERLAP = True

# Minimum ratio of unique content (vs previous chunk) to keep a chunk.
# Only rejects truly degenerate cases — intentional overlap is preserved.
VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD = 0.9

# Maximum word count for a chunk to be considered "heading-only"
VALIDATOR_HEADING_ONLY_MAX_WORDS = 3

# ==========================
# Debug Mode
# ==========================

# When True, write every produced chunk to disk for inspection
DEBUG_CHUNKS = False

# Directory for debug chunk output (relative to project root)
DEBUG_CHUNKS_DIR = "debug_chunks"

# ==========================
# Token Counter
# ==========================

# LRU cache size for token count lookups (avoids redundant tokenizer calls)
TOKEN_COUNT_CACHE_SIZE = 4096

# ==========================
# Data Directory
# ==========================

PATENT_DIRECTORY = "patents-processed"


# ==========================
# Reranker Configuration
# ==========================

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Chunks scored per batch on the reranker's GPU/MPS device. Caps peak
# memory instead of scoring every candidate chunk in one giant batch,
# which can exhaust Apple Silicon's shared MPS memory.
RERANKER_BATCH_SIZE = 16

# Number of distinct PATENTS retrieved as candidates from Qdrant.
#
# Vector search is grouped by patent_id (Qdrant group-by search), so
# this is a patent count, not a chunk count - a single patent with many
# similar chunks can't crowd other relevant patents out of the
# candidate pool the way flat top-K chunk search could. Metadata
# filtering (see app/filter_engine.py) happens AFTER this initial
# vector search, on whatever candidate patents come back - not before,
# and not as a Qdrant-side filter on the search itself. That means a
# narrower metadata filter can only ever shrink the candidate pool
# already retrieved here, never look beyond it.
PATENT_CANDIDATE_TOP_K = 100

# Number of patents returned after reranking.
#
# Applied AFTER chunks are reranked and aggregated into patents (see
# SemanticSearch.search_detailed) - not as a chunk-level truncation
# before aggregation, which could otherwise drop a patent entirely if
# none of its chunks made a flat top-K chunk cut.
FINAL_TOP_K = 10

# ==========================
# Metadata Filtering (Query Understanding LLM)
# ==========================

# Configurable local Qwen instruction model for natural-language query understanding.
# 1.5B (not the smaller 0.5B variant) so it reliably picks the right field
# code out of the FIELD_MAPPING allowlist instead of confusing similarly
# -described fields (e.g. "Publication Year" vs "Publication Country
# Code"). Chosen over 3B/4B: this machine is CPU-only with ~8GB free RAM,
# shared with the embedder and reranker models - 1.5B (~3GB in bf16) fits
# safely; 3B/4B would risk OOM/swapping once all three models are loaded.
QUERY_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# True: use the remote LLM (QUERY_LLM_REMOTE_MODEL). False: use the local
# LLM (QUERY_LLM_MODEL) instead.
USE_REMOTE_LLM = True

# QUERY_LLM_REMOTE_BASE_URL = "http://192.168.2.219:8081/v1"
QUERY_LLM_REMOTE_BASE_URL = "http://192.168.2.213:8000/v1"

# QUERY_LLM_REMOTE_MODEL = "Qwen3.6-35B-A3B-MXFP4-CRACK-MTP"
QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"

# The server does not require a real API key.
# The OpenAI client still expects a value.
QUERY_LLM_REMOTE_API_KEY = "EMPTY"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120.0

PATENT_DIRECTORY = "patents/2"

CHUNKS_COLLECTION_NAME = "patent_chunks_4096"
PATENTS_COLLECTION_NAME = "patents_metadata_4096"

# Remote embedding server
EMBEDDING_REMOTE_BASE_URL = "http://192.168.2.213:8002/v1"
EMBEDDING_REMOTE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REMOTE_API_KEY = "EMPTY"
EMBEDDING_REQUEST_TIMEOUT = 300.0
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

# Chunks per /rerank HTTP request. Mirrors EMBED_BATCH_SIZE: large
# enough to amortise network round-trip latency and keep the reranker
# server's GPU fed, without making a single request so big it dominates
# RERANKER_REQUEST_TIMEOUT on its own.
RERANK_BATCH_SIZE = 128

# How many /rerank HTTP requests to fire in parallel via a thread pool
# (mirrors EMBED_CONCURRENT_REQUESTS) - so the next batch of chunks is
# already in transit while the current one scores on the GPU, instead
# of the whole candidate pool going out as one blocking request.
RERANK_CONCURRENT_REQUESTS = 6

# The reranker model's own hard limit: query + document tokens combined
# must fit in this many tokens, or the server rejects the whole request
# with a 400 ("This model's maximum context length is 4096 tokens...").
# A chunk can be up to MAX_CHUNK_TOKENS (4096) on its own before the
# "Section: X" prefix Reranker._format_chunk adds - already at or past
# this budget before the query's own tokens are even counted - so
# Reranker truncates each formatted document to fit under
# RERANKER_MAX_CONTEXT_TOKENS minus the query's token count minus
# RERANKER_TOKEN_SAFETY_MARGIN before sending it.
RERANKER_MAX_CONTEXT_TOKENS = 4096

# Buffer subtracted from the truncation budget on top of the query's
# own token count, to absorb the small counting difference between our
# tokenizer call and whatever tokenization the server applies (special
# tokens, rounding) - keeps a truncated document from landing exactly
# on the server's limit and still getting rejected.
RERANKER_TOKEN_SAFETY_MARGIN = 16

MAX_CHUNK_TOKENS = 4096

# Safety margin subtracted from MAX_CHUNK_TOKENS when the chunker builds
# windows. TokenCounter counts chunk tokens with add_special_tokens=False,
# but the remote vLLM embedding server adds its own special token(s) per
# sequence - so a chunk that lands exactly at MAX_CHUNK_TOKENS locally can
# come in one token over the server's real limit and get rejected with
# 400 ("maximum context length is 4096 tokens ... 4097 input tokens").
# Mirrors RERANKER_TOKEN_SAFETY_MARGIN, which exists for the same reason.
EMBED_TOKEN_SAFETY_MARGIN = 8

# Number of chunks to upload to Qdrant in one request
BATCH_SIZE = 512

# Texts sent to the remote embedding server in a single HTTP request.
# With a remote GPU (DGX), this should be large to amortise network
# round-trip latency and keep the GPU fed. 256 x ~4096-token chunks is
# well within the server's batching capacity (confirmed: a single
# request of 256 max-size chunks returns 200) - the actual 400s came
# from individual chunks landing exactly at MAX_CHUNK_TOKENS, fixed via
# EMBED_TOKEN_SAFETY_MARGIN above, not from batch size.
EMBED_BATCH_SIZE = 256

# How many embedding HTTP requests to keep in flight at once. While
# batch #1 computes on the GPU, batches #2-#N are already in transit
# over the network, hiding round-trip latency. 4 is a good default;
# raise it for a high-latency link, lower it if the server is shared.
EMBED_CONCURRENT_REQUESTS = 8

# How many chunks to write between insert progress lines, counted
# across the whole run rather than per batch. The progress bar covers
# an interactive run; these lines cover a run piped to a log, where
# Rich skips the live redraw and the bar never renders. Points go to
# Qdrant BATCH_SIZE at a time, so any value at or below BATCH_SIZE
# reports every request - raise it if the insert lines crowd out the
# per-patent ones.
INSERT_REPORT_EVERY = 100

# How many patents the ingest prefetcher parses and chunks ahead of the
# embedder. Increasing this buffer keeps all CPU cores busy prefetching
# documents while the remote GPU embeds.
INGEST_PREFETCH = 512

# Raw chunks held in ingest.py's pre-embedding buffer before the very
# first embedding window is cut. 4096 == EMBED_BATCH_SIZE * 16, so the
# first round of embedding requests goes out against a full backlog
# (enough for every embed worker to have several windows queued)
# instead of firing off a handful of half-empty requests while parsing
# is still warming up. Only the first fill waits for this; every window
# after that is cut as soon as EMBED_BATCH_SIZE chunks are available.
CHUNK_QUEUE_CAPACITY = 4096

# Parallel Qdrant insert workers draining ingest.py's insertion task
# queue. Plural workers, rather than a single writer thread, so a slow
# upsert doesn't stall every batch behind it.
INSERT_WORKERS = 8

# Append-only log of patent filenames fully committed to Qdrant
# (metadata + every chunk). ingest_directory() reads it on startup to
# skip already-completed patents, so stopping and re-running the same
# command resumes instead of reprocessing from the start.
INGEST_PROGRESS_FILE = "data/ingest_progress.log"

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



QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120.0

PATENT_DIRECTORY = "patents-processed"

CHUNKS_COLLECTION_NAME = "patent_chunks_4096"
PATENTS_COLLECTION_NAME = "patents_metadata_4096"

# Remote embedding server
EMBEDDING_REMOTE_BASE_URL = "http://192.168.2.213:8002/v1"
EMBEDDING_REMOTE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REMOTE_API_KEY = "EMPTY"
EMBEDDING_REQUEST_TIMEOUT = 120.0
VECTOR_SIZE = 1024

QUERY_LLM_REMOTE_BASE_URL = "http://192.168.2.213:8000/v1"
# QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.8-27B-NVFP4"
QUERY_LLM_REMOTE_API_KEY = "EMPTY"

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

# Number of chunks to upload to Qdrant in one request
BATCH_SIZE = 512

# Texts sent to the remote embedding server in a single HTTP request.
# With a remote GPU (DGX), this should be large to amortise network
# round-trip latency and keep the GPU fed. 256 × 512-token chunks ≈
# 128 K tokens per call — well within vLLM's capacity. Raise further
# if the server has headroom; lower if requests start timing out.
EMBED_BATCH_SIZE = 256

# How many embedding HTTP requests to keep in flight at once. While
# batch #1 computes on the GPU, batches #2-#N are already in transit
# over the network, hiding round-trip latency. 4 is a good default;
# raise it for a high-latency link, lower it if the server is shared.
EMBED_CONCURRENT_REQUESTS = 4

# How many chunks to write between insert progress lines, counted
# across the whole run rather than per batch. The progress bar covers
# an interactive run; these lines cover a run piped to a log, where
# Rich skips the live redraw and the bar never renders. Points go to
# Qdrant BATCH_SIZE at a time, so any value at or below BATCH_SIZE
# reports every request - raise it if the insert lines crowd out the
# per-patent ones.
INSERT_REPORT_EVERY = 100

# Patent metadata points written to Qdrant in one request during
# ingestion. Without this each patent costs its own HTTP round trip.
METADATA_BATCH_SIZE = 64

# How many patents the ingest prefetcher parses and chunks ahead of the
# embedder. Increasing this buffer keeps all CPU cores busy prefetching
# documents while the remote GPU embeds.
INGEST_PREFETCH = 512

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

STREAMLIT_PAGE_TITLE = "Patent Semantic Search"
STREAMLIT_LAYOUT = "wide"
HISTORY_DB_PATH = "data/search_history.db"

# Number of candidate patents identified by vector search that go on
# to have EVERY one of their indexed chunks checked by the reranker
# (see app/reranker.py / SemanticSearch.search_detailed) - not a
# top-K-chunks cut, the patent's complete chunk set. Since a patent can
# have anywhere from 1 to several thousand chunks, this is the lever
# for total reranking cost per query: lower it to examine fewer
# candidate patents (each still checked completely), raise it to
# consider more candidates at higher latency cost.
PATENT_CANDIDATE_TOP_K = 300

# Chunks per candidate patent returned by the INITIAL vector-search
# step (QdrantDB.search's default group_size) - used for the "Qdrant
# Vector Search Candidates" display view and any caller that doesn't
# need more than a handful of hits per patent.
CANDIDATE_CHUNKS_PER_PATENT = 3
FINAL_TOP_K = 10


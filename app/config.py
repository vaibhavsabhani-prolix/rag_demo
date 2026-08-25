QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120.0

PATENT_DIRECTORY = "television"

CHUNKS_COLLECTION_NAME = "patent_chunks"
PATENTS_COLLECTION_NAME = "patents"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_SIZE = 1024

USE_REMOTE_LLM = True
QUERY_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QUERY_LLM_REMOTE_BASE_URL = "http://192.168.2.213:8000/v1"
QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
QUERY_LLM_REMOTE_API_KEY = "EMPTY"

USE_REMOTE_RERANKER = True
LOCAL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REMOTE_BASE_URL = "http://192.168.2.213:8001"
RERANKER_REMOTE_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REMOTE_API_KEY = "EMPTY"
RERANKER_REQUEST_TIMEOUT = 360.0

MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 20
MIN_CHUNK_WORDS = 8

# Number of chunks to upload to Qdrant in one request
BATCH_SIZE = 100

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
PATENT_CANDIDATE_TOP_K = 50

# Chunks per candidate patent returned by the INITIAL vector-search
# step only (QdrantDB.search's group_size) - used for identifying
# candidate patents and the "Qdrant Vector Search Candidates" display
# view. Reranking itself does not use this; it checks every chunk a
# candidate patent has (see PATENT_CANDIDATE_TOP_K above).
CANDIDATE_CHUNKS_PER_PATENT = 3
FINAL_TOP_K = 10

# The reranker scores every chunk of a candidate patent individually
# and takes the MAX as that patent's score (see app/reranker.py), on a
# 0-10 scale; only patents scoring at or above this are kept as a
# match. A single hard cutoff, not a tunable weighted blend.
PATENT_RELEVANCE_THRESHOLD = 7.0

assert 0.0 <= PATENT_RELEVANCE_THRESHOLD <= 10.0, (
    "PATENT_RELEVANCE_THRESHOLD must be between 0.0 and 10.0"
)

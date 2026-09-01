QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120.0

PATENT_DIRECTORY = "JP2025106030A"

CHUNKS_COLLECTION_NAME = "patent_chunks_2"
PATENTS_COLLECTION_NAME = "patents_2"

EMBEDDING_REMOTE_BASE_URL = "http://192.168.2.213:8002/v1"
EMBEDDING_REMOTE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REMOTE_API_KEY = "EMPTY"
EMBEDDING_REQUEST_TIMEOUT = 120.0
VECTOR_SIZE = 1024

QUERY_LLM_REMOTE_BASE_URL = "http://192.168.2.213:8000/v1"
QUERY_LLM_REMOTE_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
QUERY_LLM_REMOTE_API_KEY = "EMPTY"

RERANKER_REMOTE_BASE_URL = "http://192.168.2.213:8001"
RERANKER_REMOTE_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REMOTE_API_KEY = "EMPTY"
RERANKER_REQUEST_TIMEOUT = 360.0

MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 20
MIN_CHUNK_WORDS = 8

# Number of chunks to upload to Qdrant in one request
BATCH_SIZE = 100

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
METADATA_BATCH_SIZE = 256

# How many patents the ingest prefetcher parses and chunks ahead of the
# embedder. Increasing this buffer keeps all CPU cores busy prefetching
# documents while the remote GPU embeds.
INGEST_PREFETCH = 128

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


# ==============================================================
# Relevance Verification (app/relevance_verifier.py)
#
# The precision gate, running BETWEEN chunk retrieval and reranking: a
# candidate patent must literally name what the query asked for - in
# its title, abstract, or the chunks vector search matched - or it is
# dropped before the cross-encoder ever scores it. Pure string matching
# against ParsedQuery.required_phrases, which the one Query
# Understanding LLM call already produces, so this stage costs no model
# call and saves the reranker the patents it rejects.
#
# It exists because a similarity score cannot tell "a TV that is LED"
# from "an LCD TV with an LED lamp on it": both contain every word of
# "LED TV". Only the compound phrase separates them.
# ==============================================================

# Master switch. False restores the previous pipeline exactly: rerank
# everything, then cut at PATENT_RELEVANCE_THRESHOLD.
VERIFICATION_ENABLED = False

# The relevance cut applied to patents that PASSED verification. Zero
# by default: once the phrase gate has confirmed a patent literally
# names what the query asked for, the cross-encoder score's job is
# RANKING, not filtering.
#
# That is measured, not a preference. bge-reranker-v2-m3 rewards
# literal term overlap and punishes synonyms, and the phrase gate
# deliberately ACCEPTS synonyms, so the two disagree exactly where the
# gate is most useful:
#
#   query "water container" -> title "Water container"      8.9/10
#                           -> title "350ml Water bottle."  4.0/10
#   query "car"             -> title "Automobile"           3.0/10
#   query "drinking water jerrycan"
#                           -> title "350ml Water bottle."  0.0/10  (!)
#
# That last one is a correct match scored zero. A signal that returns
# 0.0 for a right answer cannot be a filter at ANY threshold - it can
# only order results that something else has already vetted. Every
# value tried here deleted correct answers: 7.0, then 5.0 (the query
# "water container" returned nothing while holding two patents titled
# "Water bottle"), then 3.0 (the jerrycan case above).
#
# Raise it only if genuinely off-topic patents start appearing - and
# fix the gate first if they do, because that is where topicality is
# decided now.
VERIFIED_RELEVANCE_THRESHOLD = 0.0

# The same cut for the BROADER fallback pass. Also zero, for the same
# reason and after the same mistake: a floor of 3.0 here deleted four
# patents titled "...Water bottle..." from a "drinking water jerrycan"
# query, all scored 0.0/10 because "jerrycan" shares no word with
# "water bottle".
#
# What keeps that pass honest is not the score but WHERE it accepts the
# relaxed wording: the patent's TITLE only (RelevanceVerifier.
# verify_broader). A shaving-razor patent whose body mentions a blade
# is not a broader match for "kitchen knife"; a patent titled "350ml
# Water bottle." is a broader match for a jerrycan query. That is a
# judgement the cross-encoder demonstrably cannot make.
BROADER_RELEVANCE_THRESHOLD = 0.0

# Keep patents that mention every required phrase, but only outside
# their title, abstract, and matching chunks (verdict RELATED), ranked
# below the confirmed matches. False is the precision-first default
# that the "LED TV" / "car" reports asked for.
VERIFICATION_KEEP_RELATED = False

# When the strict phrase gate leaves NOTHING, fall back to the relaxed
# wording Query Understanding supplies for the same query
# (ParsedQuery.fallback_phrases - the concept without its qualifier,
# e.g. "container"/"bottle"/"tank" for "water container") and show
# those separately, labelled as broader matches.
#
# Query Understanding leaves fallback_phrases EMPTY whenever dropping
# the qualifier would change what the thing IS - "LED TV" broadened to
# "television" is a different product, and that is the false positive
# this whole stage exists to prevent - so a defining qualifier
# produces no fallback tier and the answer stays honestly empty.
# Fallback results are never mixed into the exact matches.
VERIFICATION_FALLBACK_ENABLED = True

assert 0.0 <= VERIFIED_RELEVANCE_THRESHOLD <= 10.0, (
    "VERIFIED_RELEVANCE_THRESHOLD must be between 0.0 and 10.0"
)
assert 0.0 <= BROADER_RELEVANCE_THRESHOLD <= 10.0, (
    "BROADER_RELEVANCE_THRESHOLD must be between 0.0 and 10.0"
)

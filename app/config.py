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
# UI Configuration
# ==========================

# External patent detail page. {patent_id} is substituted with the
# patent's ID to build the "view patent" link in the search results UI.
PATENT_VIEW_URL_TEMPLATE = "https://www.qubeip.com/en/patent-view/{patent_id}"


# ==========================
# Reranker Configuration
# ==========================

# True = use the remote reranker server.
# False = use the local sentence-transformers CrossEncoder.
USE_REMOTE_RERANKER = True

# Local reranker model to use when USE_REMOTE_RERANKER is False.
# CPU-friendly while still giving reasonable reranking.
LOCAL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Base URL of the remote reranking server (OpenAI/TEI-style /rerank endpoint).
RERANKER_REMOTE_BASE_URL = "http://192.168.2.213:8001"

# Model name the remote reranking server expects.
RERANKER_REMOTE_MODEL = "BAAI/bge-reranker-v2-m3"

# The server does not require a real API key.
RERANKER_REMOTE_API_KEY = "EMPTY"

# Seconds to wait for the remote reranker's response. A single call can
# carry every chunk from up to PATENT_CANDIDATE_TOP_K patents (unbounded
# per patent), which can take well over a minute to score in one batch -
# so this needs more headroom than a typical HTTP call.
RERANKER_REQUEST_TIMEOUT = 360.0

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

# Number of chunks per candidate patent returned by the initial vector
# search (Qdrant group-by search group_size), used as the reranking
# pool. Caps reranker cost/latency per patent while still letting a
# patent's best-matching chunk win on something other than its single
# top vector hit.
CANDIDATE_CHUNKS_PER_PATENT = 3

# Number of patents returned after reranking.
#
# Applied AFTER chunks are reranked and aggregated into patents (see
# SemanticSearch.search_detailed) - not as a chunk-level truncation
# before aggregation, which could otherwise drop a patent entirely if
# none of its chunks made a flat top-K chunk cut.
FINAL_TOP_K = 10

# ==========================
# Reranker Blend Weights
#
# Bounds applied by app/reranker.py's _compute_weights() to the
# semantic/structured/relationship/optimization/lexical/exact_match
# blend. The LLM's ParsedQuery.ranking_weights is read as an advisory
# signal only where a dedicated field exists (relationship_satisfaction)
# - these constants are what actually enforce "semantic relevance must
# remain dominant," never trusting LLM-generated floats directly.
# ==========================

# Minimum combined weight share for semantic_score (original query) +
# structured_score (composite requirements sentence). Enforced
# regardless of what the LLM's ranking_weights say.
MIN_SEMANTIC_WEIGHT = 0.55

# Maximum weight share any single secondary category (relationship,
# optimization) may receive.
MAX_SECONDARY_WEIGHT = 0.20

# Maximum weight share for the "weak supporting" signals (lexical
# keyword coverage, exact match) - these must never meaningfully
# compete with semantic relevance.
MAX_WEAK_SIGNAL_WEIGHT = 0.10

# Fixed split of the combined semantic+structured weight budget between
# semantic_score and structured_score - never LLM-controlled, since
# structured_score is a secondary corroborating signal, not a parallel
# primary one. 0.7 means semantic_score gets 70% of that combined share.
SEM_STRUCT_SPLIT_RATIO = 0.7

# Fraction of final_score subtracted per fully-confirmed exclusion
# match. A soft penalty, never a hard filter, by default (see
# EXCLUSION_HARD_FILTER_THRESHOLD) - an excluded patent should rank
# near the bottom, not silently vanish from results.
EXCLUSION_PENALTY_WEIGHT = 0.5

# If set to a 0..1 confidence value, a candidate whose exclusion
# evidence meets/exceeds this threshold is dropped outright instead of
# merely penalized. Disabled (None) by default for two independent
# reasons:
# 1. Hard filtering risks silently dropping a relevant patent on a
#    false-positive exclusion match (e.g. a negated mention like
#    "free of lithium").
# 2. Exclusion evidence is only ever computed during the fine stage
#    (see RERANK_FINE_STAGE_TOP_N) - a candidate outside that subset
#    has no exclusion_penalty at all (defaults to 0.0), not because it
#    was cleared of the exclusion, but because it was never evaluated.
#    Enabling hard filtering today would filter fine-stage candidates
#    on real evidence while silently passing everyone else through
#    unevaluated - inconsistent filtering behavior. Do not enable this
#    unless exclusion evaluation is extended to every candidate that
#    could be hard-filtered, not just the fine-stage subset.
EXCLUSION_HARD_FILTER_THRESHOLD = None

# Multiplier floor applied to a fine-stage chunk's final_score based on how
# well it covers the query's already-extracted structure (structured_score +
# relationship_score - see Reranker._blend_and_sort). A chunk with ZERO
# measured structured/relationship coverage has its blended score multiplied
# by this floor; full coverage (=1.0) leaves the score unchanged. This can
# only shrink a score, never boost one - it exists so a chunk that is only
# broadly/genuinely on-topic can't outrank a chunk that actually satisfies
# the query's complete required structure (multiple required concepts,
# goals, constraints, relationships, or a problem->solution combination)
# purely on raw semantic similarity, which the additive weight caps above
# (MAX_SECONDARY_WEIGHT, MIN_SEMANTIC_WEIGHT) otherwise allow. Only applies
# when the query actually has a structured sentence and/or relationships to
# check coverage against - a plain-topic query with neither is unaffected
# (multiplier stays 1.0).
STRUCTURE_COVERAGE_MIN_MULTIPLIER = 0.7

# Number of top coarse-semantic-scored candidate chunks that receive
# the full "fine stage" evaluation (composite/relationship/
# optimization/exclusion synthetic-query scoring). Bounds CPU cost
# regardless of how large PATENT_CANDIDATE_TOP_K is - chunks outside
# this cutoff keep their coarse-only semantic score untouched.
RERANK_FINE_STAGE_TOP_N = 40

# Safety caps on how many relationships/optimization targets/exclusions
# get their own synthetic-query reranker pass, in case a malformed LLM
# response returns an implausibly large structure.
MAX_RELATIONSHIPS_SCORED = 5
MAX_OPTIMIZATION_TARGETS_SCORED = 5
MAX_EXCLUSIONS_SCORED = 5

# Flat importance boost a required concept/goal/constraint gets over an
# equally-scored optional one, so a required item still outranks an
# optional one when only one of the two is covered by a given chunk.
REQUIRED_IMPORTANCE_BOOST = 0.2

# Minimum importance for a concept/goal/constraint to be included in
# the composite structured-requirement sentence (see
# app/reranker.py's _build_structured_sentence) - required items are
# always included regardless of this threshold.
STRUCTURED_SIGNAL_MIN_IMPORTANCE = 0.5

# ==========================
# Metadata Filtering (Query Understanding LLM)
# ==========================

# Configurable local Qwen instruction model for natural-language query understanding.
# 1.5B (not the smaller 0.5B variant) so it reliably picks the right field
# code out of the FIELD_MAPPING allowlist instead of confusing similarly
# -described fields (e.g. "Publication Year" vs "Publication Country
# Code"). Chosen over 3B/4B: this machine is CPU-only with ~8GB free RAM,
# shared with the embedder model - 1.5B (~3GB in bf16) fits safely; 3B/4B
# would risk OOM/swapping once both models are loaded (the reranker now
# runs on a remote server, so it no longer competes for local RAM).
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

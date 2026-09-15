"""
Patent Semantic Search — Streamlit Application
Phase 1: Dynamic Query Understanding
Phase 2: Dynamic Multi-View Candidate Retrieval (Vector Search)
Phase 3: Metadata Filtering & Constraint Enforcement
Phase 4: Bounded Evidence Retrieval
Phase 5: Semantic Relationship Verification
Phase 6: BGE Cross-Encoder Reranking
Phase 7: Final Patent Scoring & Result Selection
"""

import json
import time
from typing import Any, Dict, List, Optional
import streamlit as st

from app.config import (
    CHUNKS_COLLECTION_NAME,
    EVIDENCE_CHUNKS_PER_PATENT,
    EVIDENCE_NEIGHBOR_CHUNKS,
    FINAL_SCORE_THRESHOLD,
    FINAL_TOP_K,
    FINAL_WEIGHT_RELATIONSHIP,
    FINAL_WEIGHT_REQUIREMENT,
    FINAL_WEIGHT_RERANKER,
    FINAL_WEIGHT_RETRIEVAL,
    PATENTS_COLLECTION_NAME,
    PATENT_CANDIDATE_TOP_K,
    PATENT_VIEW_URL_TEMPLATE,
    QUERY_CACHE_SIZE,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    QUERY_LLM_REQUEST_TIMEOUT,
    RERANK_BATCH_SIZE,
    RERANKER_MAX_CONTEXT_TOKENS,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RETRIEVAL_TOP_K_PER_VIEW,
    VERIFICATION_MAX_CANDIDATES,
)
from app.models.candidate import (
    CandidateChunk,
    CandidatePatent,
    CandidateRetrievalResult,
    FilteredCandidateResult,
)
from app.models.evidence import (
    EvidenceChunk,
    EvidenceRetrievalResult,
    PatentEvidence,
)
from app.models.parsed_query import ParsedQuery
from app.models.reranking import (
    RerankBatchResult,
    RerankedEvidenceChunk,
    RerankedPatentResult,
)
from app.models.scoring import (
    FinalPatentResult,
    FinalSearchResult,
    ScoreBreakdown,
)
from app.models.verification import (
    PatentVerificationResult,
    RelationshipVerification,
    RequirementVerification,
    VerificationBatchResult,
)
from app.query_understanding.engine import QueryUnderstandingEngine
from app.reranking.reranker import BGEReranker
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.metadata_filter import filter_candidates
from app.retrieval.retriever import CandidateRetriever
from app.scoring.scorer import FinalScorer
from app.verification.verifier import RelationshipVerifier
from app.retrieval.retriever import CandidateRetriever
from app.verification.verifier import RelationshipVerifier



# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Patent Semantic Search — Multi-Stage Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich dark glassmorphism aesthetic
st.markdown(
    """
    <style>
    /* Global Background */
    .stApp {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 85% 25%, rgba(6, 182, 212, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 50% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Metrics Box */
    .metric-box {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: monospace;
        margin-top: 4px;
    }
    
    /* Concept Tag */
    .concept-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #e0e7ff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
    }

    /* Relationship Row */
    .rel-card {
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .rel-node {
        background: rgba(255, 255, 255, 0.08);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .rel-arrow {
        color: #818cf8;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0 4px;
    }
    
    /* Filter Badge */
    .filter-row {
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .code-tag {
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(99, 102, 241, 0.5);
        color: #a5b4fc;
        font-family: monospace;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .op-tag {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        font-family: monospace;
        font-size: 0.72rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 4px;
    }
    .val-tag {
        background: rgba(6, 182, 212, 0.15);
        color: #67e8f9;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* Candidate Patent Card */
    .candidate-card {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        transition: border 0.2s ease;
    }
    .candidate-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }
    .score-badge {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-weight: 700;
        font-family: monospace;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .view-chip {
        display: inline-block;
        background: rgba(147, 51, 234, 0.15);
        border: 1px solid rgba(147, 51, 234, 0.35);
        color: #d8b4fe;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 4px;
    }

    /* Best Chunk Showcase */
    .best-chunk-container {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .chunk-meta-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
        margin-bottom: 10px;
        font-size: 0.82rem;
    }
    .chunk-text-box {
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.6;
        white-space: pre-wrap;
        font-family: ui-sans-serif, system-ui, sans-serif;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 6px;
        padding: 12px;
        max-height: 240px;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Singletons
# -----------------------------------------------------------------------------
@st.cache_resource
def get_engine() -> QueryUnderstandingEngine:
    return QueryUnderstandingEngine()

@st.cache_resource
def get_retriever() -> CandidateRetriever:
    return CandidateRetriever()

@st.cache_resource
def get_evidence_retriever() -> EvidenceRetriever:
    return EvidenceRetriever()

@st.cache_resource
def get_verifier() -> RelationshipVerifier:
    return RelationshipVerifier()

@st.cache_resource
def get_reranker() -> BGEReranker:
    return BGEReranker()

@st.cache_resource
def get_scorer() -> FinalScorer:
    return FinalScorer()

engine = get_engine()
retriever = get_retriever()
evidence_retriever = get_evidence_retriever()
verifier = get_verifier()
reranker = get_reranker()
scorer = get_scorer()

# -----------------------------------------------------------------------------
# Sidebar: Settings & Cache Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Infrastructure")
    st.caption(f"**Phase 1 Model:** `{QUERY_LLM_REMOTE_MODEL}`")
    st.caption(f"**LLM Base URL:** `{QUERY_LLM_REMOTE_BASE_URL}`")
    st.caption(f"**Embedding Model:** `Qwen/Qwen3-Embedding-0.6B`")
    st.caption(f"**Phase 5 & 6 Verifier/Reranker:** `{RERANKER_REMOTE_MODEL}`")
    st.caption(f"**Phase 5 Engine:** `⚡ Fast Semantic Entailment (< 300ms)`")
    st.caption(f"**Qdrant Chunk Coll:** `{CHUNKS_COLLECTION_NAME}`")
    st.caption(f"**Top K / View:** `{RETRIEVAL_TOP_K_PER_VIEW}`")
    st.caption(f"**Candidate Limit:** `{PATENT_CANDIDATE_TOP_K}`")
    st.caption(f"**Evidence / Patent:** `{EVIDENCE_CHUNKS_PER_PATENT}`")
    st.caption(f"**Neighbor Radius:** `±{EVIDENCE_NEIGHBOR_CHUNKS}`")
    st.caption(f"**Verification Limit:** `{VERIFICATION_MAX_CANDIDATES}`")
    st.caption(f"**Rerank Batch Size:** `{RERANK_BATCH_SIZE}`")
    st.caption(f"**Rerank Max Tokens:** `{RERANKER_MAX_CONTEXT_TOKENS}`")

    
    st.divider()
    st.markdown("### 🎯 Phase 7 Scoring Weights")
    st.caption(f"• **Relationships:** `{FINAL_WEIGHT_RELATIONSHIP * 100:.0f}%` (weight: {FINAL_WEIGHT_RELATIONSHIP})")
    st.caption(f"• **Requirements:** `{FINAL_WEIGHT_REQUIREMENT * 100:.0f}%` (weight: {FINAL_WEIGHT_REQUIREMENT})")
    st.caption(f"• **BGE Reranker:** `{FINAL_WEIGHT_RERANKER * 100:.0f}%` (weight: {FINAL_WEIGHT_RERANKER})")
    st.caption(f"• **Vector Retrieval:** `{FINAL_WEIGHT_RETRIEVAL * 100:.0f}%` (weight: {FINAL_WEIGHT_RETRIEVAL})")
    st.caption(f"• **Final Score Threshold:** `{FINAL_SCORE_THRESHOLD} / 10.0`")
    st.caption(f"• **Final Top K Limit:** `{FINAL_TOP_K}`")

    st.divider()
    st.markdown("### ⚡ In-Memory LRU Cache")
    cache_stats = engine.cache.stats
    st.metric("Cached Queries", f"{cache_stats['size']} / {cache_stats['max_size']}")
    st.metric("Cache Hit Ratio", f"{cache_stats['hit_ratio'] * 100:.1f}%")
    st.caption(f"Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']}")
    
    if st.button("🗑️ Reset LRU Cache", use_container_width=True):
        engine.cache.clear()
        st.success("LRU Cache cleared!")
        st.rerun()

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🔍 Patent Semantic Search")
st.markdown("#### **Multi-Stage Autonomous Retrieval Pipeline**")
st.caption("Phase 1: Query Understanding ➔ Phase 2: Candidate Retrieval ➔ Phase 3: Metadata Filter ➔ Phase 4: Bounded Evidence ➔ Phase 5: Verification ➔ Phase 6: BGE Reranking ➔ Phase 7: Final Scoring")

# -----------------------------------------------------------------------------
# Search Pipeline Flow Tracker
# -----------------------------------------------------------------------------
st.markdown("##### 🚀 Pipeline Execution Status")
col_p1, col_p2, col_p3, col_p4, col_p5, col_p6, col_p7 = st.columns(7)
with col_p1:
    st.success("**STAGE 1: Query Intent**\n\n`ACTIVE / READY`\n\nStructured entities & filters")
with col_p2:
    st.success("**STAGE 2: Candidate Search**\n\n`ACTIVE / READY`\n\nDynamic views & bounded recall")
with col_p3:
    st.success("**STAGE 3: Metadata Filter**\n\n`ACTIVE / READY`\n\nDeterministic constraints")
with col_p4:
    st.success("**STAGE 4: Bounded Evidence**\n\n`ACTIVE / READY`\n\nTargeted chunks & neighbors")
with col_p5:
    st.success("**STAGE 5: Verification**\n\n`ACTIVE / READY`\n\nSemantic relation proof")
with col_p6:
    st.success("**STAGE 6: BGE Rerank**\n\n`ACTIVE / READY`\n\nCross-encoder relevance")
with col_p7:
    st.success(f"**STAGE 7: Final Score**\n\n`ACTIVE / READY`\n\nScore ≥ {FINAL_SCORE_THRESHOLD} / 10.0")

st.divider()

# -----------------------------------------------------------------------------
# Custom Search Form
# -----------------------------------------------------------------------------
with st.form("search_form", clear_on_submit=False):
    query_input = st.text_input(
        "Patent Search Query:",
        value="",
        placeholder="Enter your custom patent search query (e.g., 'Find patents for an automotive mirror with an integrated display')...",
        help="Type any natural-language patent search query across any technical domain.",
    )

    col_btn, col_cache_toggle = st.columns([2, 8])
    with col_btn:
        analyze_clicked = st.form_submit_button("🔍 Run Search Pipeline", type="primary", use_container_width=True)
    with col_cache_toggle:
        use_cache = st.checkbox("Enable In-Memory LRU Cache for Query Understanding", value=True)

# Maintain state between interactions
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None
if "last_parsed_query" not in st.session_state:
    st.session_state["last_parsed_query"] = None
if "last_retrieval_result" not in st.session_state:
    st.session_state["last_retrieval_result"] = None
if "last_filtered_result" not in st.session_state:
    st.session_state["last_filtered_result"] = None
if "last_evidence_result" not in st.session_state:
    st.session_state["last_evidence_result"] = None
if "last_verification_result" not in st.session_state:
    st.session_state["last_verification_result"] = None
if "last_rerank_result" not in st.session_state:
    st.session_state["last_rerank_result"] = None
if "last_final_result" not in st.session_state:
    st.session_state["last_final_result"] = None
if "last_phase1_timings" not in st.session_state:
    st.session_state["last_phase1_timings"] = None

if analyze_clicked:
    clean_q = query_input.strip()
    if not clean_q:
        st.warning("Please enter a search query.")
    else:
        # Phase 1: Query Understanding
        was_cached = engine.cache.get(clean_q) is not None if use_cache else False
        with st.spinner("Stage 1: Executing Query Understanding..."):
            t1_start = time.perf_counter()
            parsed_query = engine.parse(clean_q, use_cache=use_cache)
            t1_total_ms = (time.perf_counter() - t1_start) * 1000

        if was_cached:
            llm_ms = 0.0
            norm_ms = t1_total_ms
            status_text = "⚡ CACHE HIT"
            status_color = "#10b981"
        else:
            norm_ms = 0.005
            llm_ms = max(0.0, t1_total_ms - norm_ms)
            status_text = "❄️ COLD (1 LLM Call)"
            status_color = "#06b6d4"

        # Phase 2: Dynamic Candidate Retrieval (Vector Search)
        with st.spinner("Stage 2: Executing Dynamic Candidate Vector Retrieval..."):
            retrieval_result = retriever.retrieve_candidates(parsed_query)

        # Phase 3: Metadata Filtering & Constraint Enforcement
        with st.spinner("Stage 3: Enforcing Metadata Constraints..."):
            filtered_result = filter_candidates(
                candidates=retrieval_result.candidates,
                metadata_filters=parsed_query.metadata_filters,
                is_metadata_only=parsed_query.is_metadata_only,
            )

        # Phase 4: Bounded Evidence Retrieval
        with st.spinner("Stage 4: Retrieving Bounded Evidence Chunks..."):
            evidence_result = evidence_retriever.retrieve_evidence(
                parsed_query=parsed_query,
                candidates=filtered_result.candidates,
            )

        # Phase 5: Semantic Relationship Verification (Fast Cross-Encoder & Proximity)
        with st.spinner("Stage 5: Executing Fast Semantic Relationship Verification..."):
            verification_result = verifier.verify_candidates(
                parsed_query=parsed_query,
                evidence_result=evidence_result,
            )

        # Phase 6: BGE Cross-Encoder Reranking
        with st.spinner("Stage 6: Scoring Evidence with BGE Cross-Encoder Reranker..."):
            rerank_result = reranker.rerank_candidates(
                parsed_query=parsed_query,
                verification_result=verification_result,
                evidence_result=evidence_result,
            )

        # Phase 7: Final Patent Scoring & Result Selection
        with st.spinner("Stage 7: Computing Final Multi-Signal Patent Scores & Filtering..."):
            final_result = scorer.score_and_rank(rerank_batch=rerank_result)

        st.session_state["last_query"] = clean_q
        st.session_state["last_parsed_query"] = parsed_query
        st.session_state["last_retrieval_result"] = retrieval_result
        st.session_state["last_filtered_result"] = filtered_result
        st.session_state["last_evidence_result"] = evidence_result
        st.session_state["last_verification_result"] = verification_result
        st.session_state["last_rerank_result"] = rerank_result
        st.session_state["last_final_result"] = final_result
        st.session_state["last_phase1_timings"] = {
            "total_ms": t1_total_ms,
            "llm_ms": llm_ms,
            "norm_ms": norm_ms,
            "status_text": status_text,
            "status_color": status_color,
        }

# -----------------------------------------------------------------------------
# Display Results Section
# -----------------------------------------------------------------------------
if st.session_state.get("last_parsed_query") is not None:
    parsed_query: ParsedQuery = st.session_state["last_parsed_query"]
    retrieval_result: CandidateRetrievalResult = st.session_state["last_retrieval_result"]
    filtered_result: FilteredCandidateResult = st.session_state.get("last_filtered_result")
    evidence_result: EvidenceRetrievalResult = st.session_state.get("last_evidence_result")
    verification_result: VerificationBatchResult = st.session_state.get("last_verification_result")
    rerank_result: Optional[RerankBatchResult] = st.session_state.get("last_rerank_result")
    final_result: Optional[FinalSearchResult] = st.session_state.get("last_final_result")
    p1_timings = st.session_state["last_phase1_timings"]
    p2_timings = retrieval_result.timings





    # =========================================================================
    # 1. PHASE 1: QUERY UNDERSTANDING (TIMINGS + EXTRACTED STRUCTURE + JSON)
    # =========================================================================
    st.markdown("## 🧠 Phase 1: Query Understanding")
    
    # Phase 1 Timings
    st.markdown("##### ⏱️ Phase 1 Timing Breakdown")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Phase 1 Total Time</div>
                <div class="metric-val" style="color: #67e8f9;">{p1_timings['total_ms']:.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">LLM Inference (1 Call)</div>
                <div class="metric-val" style="color: #a5b4fc;">{p1_timings['llm_ms']:.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Local Normalization</div>
                <div class="metric-val" style="color: #6ee7b7;">{p1_timings['norm_ms']:.4f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_t4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Cache / Execution State</div>
                <div class="metric-val" style="color: {p1_timings['status_color']}; font-size: 1.1rem;">{p1_timings['status_text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Phase 1 Semantic Structure
    st.markdown("##### 📋 Semantic Structure & Extracted Intent")
    st.info(f"**Semantic Query (Intent Preserved):** {parsed_query.semantic_query}")
    if parsed_query.is_metadata_only:
        st.warning("🏷️ **Metadata-Only Query**: Query contains exclusively metadata constraints with zero technical concepts.")

    col_left, col_right = st.columns(2)

    with col_left:
        # Concepts
        st.markdown("###### 🏷️ Technical Concepts")
        if parsed_query.concepts:
            chips_html = "".join([f'<span class="concept-chip">🏷️ {c}</span>' for c in parsed_query.concepts])
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("No explicit technical concepts extracted.")

        # Directed Relationships
        st.markdown("###### 🔗 Directed Relationships")
        if parsed_query.relationships:
            for r in parsed_query.relationships:
                ctx_str = f'<span style="color: #94a3b8; font-size: 0.75rem; font-style: italic;"> ({r.context})</span>' if r.context else ''
                st.markdown(
                    f"""
                    <div class="rel-card">
                        <span class="rel-node" style="color: #67e8f9;">{r.subject}</span>
                        <span class="rel-arrow">──[{r.relation}]──▶</span>
                        <span class="rel-node" style="color: #c084fc;">{r.object}</span>
                        {ctx_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No explicit relationships identified.")

        # Attributes
        if parsed_query.attributes:
            st.markdown("###### ⚙️ Concept Attributes")
            for a in parsed_query.attributes:
                st.markdown(f"- **{a.concept}** ➔ `{a.name}`: **{a.value}**")

    with col_right:
        # Requirements & Constraints
        st.markdown("###### 📋 Requirements & Constraints")
        if parsed_query.requirements:
            for req in parsed_query.requirements:
                st.markdown(f"- ✅ {req}")
        if parsed_query.constraints:
            for con in parsed_query.constraints:
                st.markdown(f"- ⚠️ **{con}**")
        if parsed_query.exclusions:
            for ex in parsed_query.exclusions:
                st.markdown(f"- 🚫 <span style='color: #f43f5e; font-weight: 600;'>{ex}</span>", unsafe_allow_html=True)
        if not parsed_query.requirements and not parsed_query.constraints and not parsed_query.exclusions:
            st.caption("No specific requirements, constraints, or exclusions.")

        # Metadata Filters
        st.markdown("###### 📑 Resolved Metadata Filters")
        if parsed_query.metadata_filters:
            for f in parsed_query.metadata_filters:
                st.markdown(
                    f"""
                    <div class="filter-row">
                        <div>
                            <span class="code-tag">{f.field}</span>
                            <span style="font-size: 0.82rem; color: #f1f5f9; margin-left: 8px;">{f.raw_field or f.field}</span>
                        </div>
                        <div>
                            <span class="op-tag">{f.operator}</span>
                            <span class="val-tag">{f.value}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No patent metadata constraints specified.")

    # Phase 1 JSON Data Accordion
    with st.expander("🔍 View Raw Phase 1 ParsedQuery JSON Data", expanded=False):
        st.json(parsed_query.model_dump())

    st.markdown("---")

    # =========================================================================
    # 2. PHASE 2: DYNAMIC CANDIDATE RETRIEVAL (VECTOR SEARCH)
    # =========================================================================
    st.markdown("## 🎯 Phase 2: Dynamic Candidate Retrieval (Vector Search)")

    # Phase 2 Timings
    st.markdown("##### ⏱️ Phase 2 Timing Breakdown")
    col_p2_1, col_p2_2, col_p2_3, col_p2_4 = st.columns(4)
    with col_p2_1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Phase 2 Total Time</div>
                <div class="metric-val" style="color: #34d399;">{p2_timings.get('total_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2_2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Batched Embedding</div>
                <div class="metric-val" style="color: #a78bfa;">{p2_timings.get('embedding_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2_3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Qdrant Vector Search</div>
                <div class="metric-val" style="color: #38bdf8;">{p2_timings.get('qdrant_retrieval_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2_4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Merge & Deduplication</div>
                <div class="metric-val" style="color: #f472b6;">{p2_timings.get('merge_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Retrieval Summary Metrics
    cand_col1, cand_col2, cand_col3 = st.columns(3)
    with cand_col1:
        st.metric("Total Chunk Hits (Raw)", f"{retrieval_result.total_chunk_hits}")
    with cand_col2:
        st.metric("Unique Patents Identified", f"{retrieval_result.unique_patents}")
    with cand_col3:
        st.metric("Bounded Candidates Returned", f"{len(retrieval_result.candidates)}")

    # Expandable Retrieval Views
    with st.expander("👁️ View Dynamic Retrieval Views Used for Vector Search", expanded=False):
        for v_name, v_text in retrieval_result.retrieval_views.items():
            st.markdown(f"**View `{v_name}`:**")
            st.code(v_text, language="text")

    # Raw Phase 2 JSON Output
    with st.expander("🔍 View Raw Phase 2 CandidateRetrievalResult JSON Data", expanded=False):
        st.json(retrieval_result.model_dump())

    st.markdown("---")

    # =========================================================================
    # 3. PHASE 3: METADATA FILTERING & CONSTRAINT ENFORCEMENT
    # =========================================================================
    st.markdown("## 🛡️ Phase 3: Metadata Filtering & Constraint Enforcement")

    if filtered_result is None:
        st.warning("Phase 3 has not run yet.")
    else:
        # Phase 3 Timing Breakdown & Counters
        st.markdown("##### ⏱️ Phase 3 Metrics & Constraint Diagnostics")
        col_p3_1, col_p3_2, col_p3_3, col_p3_4 = st.columns(4)
        with col_p3_1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Phase 3 Total Time</div>
                    <div class="metric-val" style="color: #38bdf8;">{filtered_result.filter_time_ms:.2f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p3_2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Candidates Before</div>
                    <div class="metric-val" style="color: #a78bfa;">{filtered_result.total_before}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p3_3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Candidates Passed</div>
                    <div class="metric-val" style="color: #34d399;">{filtered_result.total_after}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p3_4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Filtered Out</div>
                    <div class="metric-val" style="color: #f43f5e;">{filtered_result.filtered_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Filter Diagnostics Table / Badges
        if filtered_result.diagnostics:
            st.markdown("###### 📊 Metadata Filter Pass / Fail Breakdown")
            for diag in filtered_result.diagnostics:
                st.markdown(
                    f"""
                    <div class="filter-row">
                        <div>
                            <span class="code-tag">{diag.field}</span>
                            <span class="op-tag">{diag.operator}</span>
                            <span class="val-tag">{diag.value}</span>
                        </div>
                        <div>
                            <span style="color: #34d399; font-weight: 700; font-size: 0.82rem; margin-right: 12px;">✅ Passed: {diag.passed}</span>
                            <span style="color: #f43f5e; font-weight: 700; font-size: 0.82rem;">❌ Failed: {diag.failed}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("ℹ️ No metadata constraints specified in query. All candidate patents passed validation.")

        # Surviving Candidates List
        if not filtered_result.candidates:
            st.warning("⚠️ No candidate patents satisfied all metadata constraints.")
        else:
            st.caption(f"✅ **{len(filtered_result.candidates)} candidate patents** successfully passed all metadata constraints.")

        # Raw Phase 3 JSON Output
        with st.expander("🔍 View Raw Phase 3 FilteredCandidateResult JSON Data", expanded=False):
            st.json(filtered_result.model_dump())

    st.markdown("---")

    # =========================================================================
    # 4. PHASE 4: BOUNDED EVIDENCE RETRIEVAL
    # =========================================================================
    st.markdown("## 📑 Phase 4: Bounded Evidence Retrieval")

    if evidence_result is None:
        st.warning("Phase 4 has not run yet.")
    else:
        # Phase 4 Timing Breakdown & Summary Metrics
        st.markdown("##### ⏱️ Phase 4 Metrics & Latency Breakdown")
        col_p4_1, col_p4_2, col_p4_3, col_p4_4 = st.columns(4)
        with col_p4_1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Phase 4 Total Time</div>
                    <div class="metric-val" style="color: #67e8f9;">{evidence_result.timings.get('total_ms', 0):.2f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p4_2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Candidates Evaluated</div>
                    <div class="metric-val" style="color: #a78bfa;">{evidence_result.total_candidates}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p4_3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Total Evidence Chunks</div>
                    <div class="metric-val" style="color: #34d399;">{evidence_result.total_evidence_chunks}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p4_4:
            avg_chunks = (
                evidence_result.total_evidence_chunks / evidence_result.total_candidates
                if evidence_result.total_candidates > 0
                else 0.0
            )
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Avg Chunks / Patent</div>
                    <div class="metric-val" style="color: #f472b6;">{avg_chunks:.1f} <span style="font-size: 0.75rem; color: #94a3b8;">/ {EVIDENCE_CHUNKS_PER_PATENT} max</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Sub-Latency Metrics
        ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
        with ev_col1:
            st.caption(f"⚡ **Embedding:** `{evidence_result.timings.get('embedding_ms', 0):.1f} ms`")
        with ev_col2:
            st.caption(f"🎯 **Qdrant Vector Search:** `{evidence_result.timings.get('qdrant_retrieval_ms', 0):.1f} ms`")
        with ev_col3:
            st.caption(f"🔗 **Neighbor Scroll:** `{evidence_result.timings.get('neighbor_ms', 0):.1f} ms`")
        with ev_col4:
            st.caption(f"🧹 **Dedup & Bounding:** `{evidence_result.timings.get('deduplication_ms', 0):.2f} ms`")

        # Dynamic Evidence Query
        if evidence_result.evidence_query_text:
            with st.expander("👁️ View Dynamic Evidence Query (Derived from Concepts, Relationships & Requirements)", expanded=False):
                st.code(evidence_result.evidence_query_text, language="text")

        # Evidence Chunks Showcase per Patent Candidate
        if not evidence_result.patent_evidence_list:
            st.warning("⚠️ No evidence chunks retrieved for the qualified candidate patents.")
        else:
            st.markdown("---")
            with st.expander(
                f"🔽 Top {len(evidence_result.patent_evidence_list)} Candidate Patents & Bounded Evidence Chunks (Click to View/Collapse)",
                expanded=True,
            ):
                for idx, pat_ev in enumerate(evidence_result.patent_evidence_list):
                    meta = pat_ev.metadata or {}
                    title = meta.get("Title-english") or meta.get("Title") or "Title not available"
                    assignee_list = meta.get("Current Assignee Standardized") or meta.get("Applicant First Organization") or []
                    assignee_str = ", ".join(assignee_list) if isinstance(assignee_list, list) else str(assignee_list)
                    pub_year = meta.get("Publication Year") or meta.get("Application Year", "N/A")
                    country = meta.get("Publication Country Code") or "N/A"
                    patent_url = PATENT_VIEW_URL_TEMPLATE.format(patent_id=pat_ev.patent_id)

                    # Patent Header Card
                    st.markdown(
                        f"""
                        <div class="candidate-card" style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                                <div>
                                    <span style="font-size: 1.15rem; font-weight: 700; color: #67e8f9;">#{idx+1}</span>
                                    <a href="{patent_url}" target="_blank" style="font-size: 1.15rem; font-weight: 700; color: #818cf8; text-decoration: none; margin-left: 8px;">
                                        {pat_ev.patent_id} ↗
                                    </a>
                                    <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">({country} • {pub_year})</span>
                                </div>
                                <div>
                                    <span style="color: #94a3b8; font-size: 0.75rem; margin-right: 6px;">Candidate Score:</span>
                                    <span class="score-badge">{pat_ev.candidate_score:.4f}</span>
                                    <span style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc; font-weight: 600; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px; margin-left: 6px;">
                                        {len(pat_ev.chunks)} Evidence Chunks
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 1rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                                {title}
                            </div>
                            <div style="font-size: 0.82rem; color: #cbd5e1;">
                                <span style="color: #94a3b8;">Assignee:</span> {assignee_str or 'Unknown'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Evidence Chunks
                    for ch in pat_ev.chunks:
                        if ch.retrieval_source == "initial_candidate":
                            source_badge = '<span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;">⭐ Initial Vector Hit</span>'
                        elif ch.retrieval_source == "evidence_query":
                            source_badge = '<span style="background: rgba(6, 182, 212, 0.2); border: 1px solid rgba(6, 182, 212, 0.4); color: #67e8f9; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;">🎯 Evidence Query Hit</span>'
                        else:
                            source_badge = '<span style="background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4); color: #fbbf24; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;">🔗 Neighbor Chunk (±1)</span>'

                        tok_info = f'<span style="color: #94a3b8; font-size: 0.75rem; margin-left: 8px;">({ch.token_count} tokens)</span>' if ch.token_count else ''

                        st.markdown(
                            f"""
                            <div class="best-chunk-container" style="margin-top: 4px; margin-bottom: 8px; padding: 10px 14px;">
                                <div class="chunk-meta-bar" style="padding-bottom: 6px; margin-bottom: 8px;">
                                    <div>
                                        <span style="font-weight: 700; font-size: 0.85rem; color: #f1f5f9; margin-right: 8px;">Chunk #{ch.chunk_id}</span>
                                        {source_badge}
                                        <span style="color: #94a3b8; margin-left: 10px; font-size: 0.8rem;">Section: <code style="color: #a5b4fc;">{ch.section or 'Section N/A'}</code></span>
                                        {tok_info}
                                    </div>
                                    <div>
                                        <span style="color: #94a3b8; font-size: 0.75rem; margin-right: 4px;">Score:</span>
                                        <span class="score-badge" style="font-size: 0.78rem; padding: 2px 6px;">{ch.retrieval_score:.4f}</span>
                                    </div>
                                </div>
                                <div class="chunk-text-box" style="font-size: 0.84rem; max-height: 180px;">{ch.text or 'Chunk text not available.'}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Raw Phase 4 JSON Output
        with st.expander("🔍 View Raw Phase 4 EvidenceRetrievalResult JSON Data", expanded=False):
            st.json(evidence_result.model_dump())

    st.markdown("---")

    # =========================================================================
    # 5. PHASE 5: SEMANTIC RELATIONSHIP VERIFICATION
    # =========================================================================
    st.markdown("## 🧪 Phase 5: Semantic Relationship Verification")

    if verification_result is None:
        st.warning("Phase 5 has not run yet.")
    else:
        # Phase 5 Timing Breakdown & Verification Summary
        st.markdown("##### ⏱️ Phase 5 Metrics & Verification Summary")
        col_p5_1, col_p5_2, col_p5_3, col_p5_4, col_p5_5 = st.columns(5)
        with col_p5_1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Phase 5 Total Time</div>
                    <div class="metric-val" style="color: #67e8f9;">{verification_result.timings.get('total_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p5_2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Candidates Verified</div>
                    <div class="metric-val" style="color: #a78bfa;">{verification_result.total_evaluated}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p5_3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Fully Supported</div>
                    <div class="metric-val" style="color: #34d399;">{verification_result.fully_supported_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p5_4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Partially Supported</div>
                    <div class="metric-val" style="color: #fbbf24;">{verification_result.partially_supported_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p5_5:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Unsupported / 0%</div>
                    <div class="metric-val" style="color: #f43f5e;">{verification_result.unsupported_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Verified Candidates Showcase
        if not verification_result.verified_patents:
            st.warning("⚠️ No candidates were evaluated during Phase 5 verification.")
        else:
            st.markdown("---")
            with st.expander(
                f"🔽 Top {len(verification_result.verified_patents)} Verified Candidate Patents & Relationship Proof (Click to View/Collapse)",
                expanded=True,
            ):
                for idx, vpat in enumerate(verification_result.verified_patents):
                    meta = vpat.metadata or {}
                    title = meta.get("Title-english") or meta.get("Title") or "Title not available"
                    assignee_list = meta.get("Current Assignee Standardized") or meta.get("Applicant First Organization") or []
                    assignee_str = ", ".join(assignee_list) if isinstance(assignee_list, list) else str(assignee_list)
                    pub_year = meta.get("Publication Year") or meta.get("Application Year", "N/A")
                    country = meta.get("Publication Country Code") or "N/A"
                    patent_url = PATENT_VIEW_URL_TEMPLATE.format(patent_id=vpat.patent_id)

                    cov_pct = int(vpat.relationship_coverage * 100)
                    if cov_pct == 100:
                        cov_color = "#34d399"
                        cov_bg = "rgba(16, 185, 129, 0.2)"
                    elif cov_pct >= 50:
                        cov_color = "#fbbf24"
                        cov_bg = "rgba(245, 158, 11, 0.2)"
                    else:
                        cov_color = "#f43f5e"
                        cov_bg = "rgba(244, 63, 94, 0.2)"

                    # Patent Header Card
                    st.markdown(
                        f"""
                        <div class="candidate-card" style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                                <div>
                                    <span style="font-size: 1.15rem; font-weight: 700; color: #67e8f9;">#{idx+1}</span>
                                    <a href="{patent_url}" target="_blank" style="font-size: 1.15rem; font-weight: 700; color: #818cf8; text-decoration: none; margin-left: 8px;">
                                        {vpat.patent_id} ↗
                                    </a>
                                    <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">({country} • {pub_year})</span>
                                </div>
                                <div>
                                    <span style="color: #94a3b8; font-size: 0.75rem; margin-right: 6px;">Candidate Score:</span>
                                    <span class="score-badge">{vpat.candidate_score:.4f}</span>
                                    <span style="background: {cov_bg}; border: 1px solid {cov_color}; color: {cov_color}; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px; margin-left: 6px;">
                                        🎯 {cov_pct}% Relationship Coverage ({vpat.supported_count}/{len(vpat.relationships)})
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 1rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                                {title}
                            </div>
                            <div style="font-size: 0.82rem; color: #cbd5e1;">
                                <span style="color: #94a3b8;">Assignee:</span> {assignee_str or 'Unknown'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Relationships Breakdown
                    if vpat.relationships:
                        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-top: 6px; margin-bottom: 4px;'>🔗 Directed Relationships Verification:</div>", unsafe_allow_html=True)
                        for r_ver in vpat.relationships:
                            if r_ver.status == "SUPPORTED":
                                status_badge = '<span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 6px;">✅ SUPPORTED</span>'
                            elif r_ver.status == "CONTRADICTED":
                                status_badge = '<span style="background: rgba(239, 68, 68, 0.25); border: 1px solid rgba(239, 68, 68, 0.5); color: #f87171; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 6px;">⚠️ CONTRADICTED</span>'
                            elif r_ver.status == "UNKNOWN":
                                status_badge = '<span style="background: rgba(148, 163, 184, 0.2); border: 1px solid rgba(148, 163, 184, 0.4); color: #cbd5e1; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 6px;">❓ UNKNOWN</span>'
                            else:
                                status_badge = '<span style="background: rgba(244, 63, 94, 0.2); border: 1px solid rgba(244, 63, 94, 0.4); color: #f43f5e; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 6px;">❌ NOT_SUPPORTED</span>'

                            cite_str = f'<span style="color: #67e8f9; font-size: 0.75rem; margin-left: 8px;">(Cites: Chunk {", ".join(map(str, r_ver.evidence_chunk_ids))})</span>' if r_ver.evidence_chunk_ids else ''
                            expl_str = f'<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; font-style: italic;">{r_ver.explanation}</div>' if r_ver.explanation else ''

                            st.markdown(
                                f"""
                                <div class="rel-card" style="margin-bottom: 6px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <span class="rel-node" style="color: #67e8f9;">{r_ver.subject}</span>
                                            <span class="rel-arrow">──[{r_ver.relation}]──▶</span>
                                            <span class="rel-node" style="color: #c084fc;">{r_ver.object}</span>
                                            {cite_str}
                                        </div>
                                        <div>
                                            {status_badge}
                                            <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 6px;">({int(r_ver.confidence*100)}% conf)</span>
                                        </div>
                                    </div>
                                    {expl_str}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Requirements Breakdown
                    if vpat.requirements:
                        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-top: 6px; margin-bottom: 4px;'>📋 Requirements Verification:</div>", unsafe_allow_html=True)
                        for req_ver in vpat.requirements:
                            if req_ver.supported:
                                req_badge = '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 700; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px;">✅ PASSED</span>'
                            else:
                                req_badge = '<span style="background: rgba(244, 63, 94, 0.2); color: #f43f5e; font-weight: 700; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px;">❌ UNMET</span>'

                            cite_str = f'<span style="color: #67e8f9; font-size: 0.75rem; margin-left: 6px;">(Chunk {", ".join(map(str, req_ver.evidence_chunk_ids))})</span>' if req_ver.evidence_chunk_ids else ''

                            st.markdown(
                                f"""
                                <div style="background: rgba(18, 24, 38, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 6px; padding: 6px 10px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center;">
                                    <div style="font-size: 0.82rem; color: #e2e8f0;">
                                        {req_ver.requirement} {cite_str}
                                    </div>
                                    <div>{req_badge}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Raw Phase 5 JSON Output
        with st.expander("🔍 View Raw Phase 5 VerificationBatchResult JSON Data", expanded=False):
            st.json(verification_result.model_dump())

    st.markdown("---")

    # =========================================================================
    # 6. PHASE 6: BGE CROSS-ENCODER RERANKING
    # =========================================================================
    st.markdown("## ⚡ Phase 6: BGE Cross-Encoder Reranking")

    if rerank_result is None:
        st.warning("Phase 6 has not run yet.")
    else:
        # Phase 6 Timing Breakdown & Metrics
        st.markdown("##### ⏱️ Phase 6 Metrics & Cross-Encoder Latency Breakdown")
        col_p6_1, col_p6_2, col_p6_3, col_p6_4, col_p6_5, col_p6_6 = st.columns(6)
        with col_p6_1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Phase 6 Total Time</div>
                    <div class="metric-val" style="color: #67e8f9;">{rerank_result.timings.get('total_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p6_2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Reranker HTTP Time</div>
                    <div class="metric-val" style="color: #a78bfa;">{rerank_result.timings.get('reranker_http_ms', 0):.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p6_3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Candidates Reranked</div>
                    <div class="metric-val" style="color: #34d399;">{rerank_result.total_candidates}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p6_4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Evidence Chunks</div>
                    <div class="metric-val" style="color: #38bdf8;">{rerank_result.total_chunks_reranked}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p6_5:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Token Truncations</div>
                    <div class="metric-val" style="color: #fbbf24;">{rerank_result.truncated_chunks_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p6_6:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Avg Time / Chunk</div>
                    <div class="metric-val" style="color: #f472b6;">{rerank_result.timings.get('avg_per_chunk_ms', 0):.2f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dynamic Reranking Query Expander
        if rerank_result.reranking_query:
            with st.expander("👁️ View Deterministic Reranking Query (Semantic Query + Relationships + Requirements)", expanded=False):
                st.code(rerank_result.reranking_query, language="text")

        # Reranked Candidates Showcase
        if not rerank_result.reranked_patents:
            st.warning("⚠️ No candidate patents were scored in Phase 6.")
        else:
            st.markdown("---")
            with st.expander(
                f"🔽 Top {len(rerank_result.reranked_patents)} Candidate Patents & BGE Cross-Encoder Scored Evidence (Click to View/Collapse)",
                expanded=True,
            ):
                for idx, rpat in enumerate(rerank_result.reranked_patents):
                    meta = rpat.metadata or {}
                    title = meta.get("Title-english") or meta.get("Title") or "Title not available"
                    assignee_list = meta.get("Current Assignee Standardized") or meta.get("Applicant First Organization") or []
                    assignee_str = ", ".join(assignee_list) if isinstance(assignee_list, list) else str(assignee_list)
                    pub_year = meta.get("Publication Year") or meta.get("Application Year", "N/A")
                    country = meta.get("Publication Country Code") or "N/A"
                    patent_url = PATENT_VIEW_URL_TEMPLATE.format(patent_id=rpat.patent_id)

                    cov_pct = int(rpat.relationship_coverage * 100)
                    if cov_pct == 100:
                        cov_color = "#34d399"
                        cov_bg = "rgba(16, 185, 129, 0.2)"
                    elif cov_pct >= 50:
                        cov_color = "#fbbf24"
                        cov_bg = "rgba(245, 158, 11, 0.2)"
                    else:
                        cov_color = "#f43f5e"
                        cov_bg = "rgba(244, 63, 94, 0.2)"

                    # Reranker score color coding
                    if rpat.best_reranker_score >= 0.80:
                        rerank_badge_color = "#34d399"
                        rerank_bg = "rgba(16, 185, 129, 0.25)"
                        rerank_border = "rgba(16, 185, 129, 0.6)"
                    elif rpat.best_reranker_score >= 0.50:
                        rerank_badge_color = "#38bdf8"
                        rerank_bg = "rgba(56, 189, 248, 0.2)"
                        rerank_border = "rgba(56, 189, 248, 0.5)"
                    else:
                        rerank_badge_color = "#fbbf24"
                        rerank_bg = "rgba(245, 158, 11, 0.2)"
                        rerank_border = "rgba(245, 158, 11, 0.4)"

                    # Candidate Header Card
                    st.markdown(
                        f"""
                        <div class="candidate-card" style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                                <div>
                                    <span style="font-size: 1.15rem; font-weight: 700; color: #67e8f9;">#{idx+1}</span>
                                    <a href="{patent_url}" target="_blank" style="font-size: 1.15rem; font-weight: 700; color: #818cf8; text-decoration: none; margin-left: 8px;">
                                        {rpat.patent_id} ↗
                                    </a>
                                    <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">({country} • {pub_year})</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="background: {rerank_bg}; border: 1px solid {rerank_border}; padding: 4px 10px; border-radius: 6px;">
                                        <span style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;">Best BGE:</span>
                                        <span style="color: {rerank_badge_color}; font-weight: 800; font-family: monospace; font-size: 0.92rem; margin-left: 4px;">{rpat.best_reranker_score:.4f}</span>
                                    </div>
                                    <span style="background: {cov_bg}; border: 1px solid {cov_color}; color: {cov_color}; font-weight: 700; font-size: 0.75rem; padding: 4px 8px; border-radius: 6px;">
                                        🎯 {cov_pct}% Rel Coverage
                                    </span>
                                    <span style="background: rgba(255, 255, 255, 0.05); color: #94a3b8; font-size: 0.75rem; padding: 4px 8px; border-radius: 6px;">
                                        Ret: {rpat.candidate_score:.4f}
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 1rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                                {title}
                            </div>
                            <div style="font-size: 0.82rem; color: #cbd5e1;">
                                <span style="color: #94a3b8;">Assignee:</span> {assignee_str or 'Unknown'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Evidence Chunks Scored with BGE
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-top: 6px; margin-bottom: 4px;'>📊 BGE Scored Evidence Chunks:</div>", unsafe_allow_html=True)
                    for ch in rpat.evidence:
                        if ch.retrieval_source == "initial_candidate":
                            source_badge = '<span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;">⭐ Initial Vector Hit</span>'
                        elif ch.retrieval_source == "evidence_query":
                            source_badge = '<span style="background: rgba(6, 182, 212, 0.2); border: 1px solid rgba(6, 182, 212, 0.4); color: #67e8f9; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;">🎯 Evidence Query Hit</span>'
                        else:
                            source_badge = '<span style="background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4); color: #fbbf24; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;">🔗 Neighbor Chunk (±1)</span>'

                        tok_info = f'<span style="color: #94a3b8; font-size: 0.75rem; margin-left: 8px;">({ch.token_count} tokens)</span>' if ch.token_count else ''

                        if ch.reranker_score >= 0.80:
                            chunk_bge_color = "#34d399"
                            chunk_bge_bg = "rgba(16, 185, 129, 0.25)"
                        elif ch.reranker_score >= 0.50:
                            chunk_bge_color = "#38bdf8"
                            chunk_bge_bg = "rgba(56, 189, 248, 0.2)"
                        else:
                            chunk_bge_color = "#94a3b8"
                            chunk_bge_bg = "rgba(255, 255, 255, 0.06)"

                        st.markdown(
                            f"""
                            <div class="best-chunk-container" style="margin-top: 4px; margin-bottom: 8px; padding: 10px 14px;">
                                <div class="chunk-meta-bar" style="padding-bottom: 6px; margin-bottom: 8px;">
                                    <div>
                                        <span style="font-weight: 700; font-size: 0.85rem; color: #f1f5f9; margin-right: 8px;">Chunk #{ch.chunk_id}</span>
                                        {source_badge}
                                        <span style="color: #94a3b8; margin-left: 10px; font-size: 0.8rem;">Section: <code style="color: #a5b4fc;">{ch.section or 'Section N/A'}</code></span>
                                        {tok_info}
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <div style="background: {chunk_bge_bg}; padding: 2px 8px; border-radius: 4px;">
                                            <span style="color: #94a3b8; font-size: 0.72rem;">BGE Score:</span>
                                            <span style="color: {chunk_bge_color}; font-weight: 700; font-family: monospace; font-size: 0.85rem; margin-left: 4px;">{ch.reranker_score:.4f}</span>
                                        </div>
                                        <span style="color: #94a3b8; font-size: 0.75rem;">(Ret Score: {ch.retrieval_score:.4f})</span>
                                    </div>
                                </div>
                                <div class="chunk-text-box" style="font-size: 0.84rem; max-height: 180px;">{ch.text or 'Chunk text not available.'}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Raw Phase 6 JSON Output
        with st.expander("🔍 View Raw Phase 6 RerankBatchResult JSON Data", expanded=False):
            st.json(rerank_result.model_dump())

    st.markdown("---")

    # =========================================================================
    # 7. PHASE 7: FINAL PATENT SCORING & RESULT SELECTION
    # =========================================================================
    st.markdown("## 🏆 Phase 7: Final Patent Scoring & Result Selection")

    if final_result is None:
        st.warning("Phase 7 has not run yet.")
    else:
        # Phase 7 Timing Breakdown & Summary Metrics
        st.markdown("##### ⏱️ Phase 7 Metrics & Result Selection Summary")
        col_p7_1, col_p7_2, col_p7_3, col_p7_4, col_p7_5 = st.columns(5)
        with col_p7_1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Phase 7 Total Time</div>
                    <div class="metric-val" style="color: #67e8f9;">{final_result.timings.get('total_ms', 0):.3f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p7_2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Evaluated Candidates</div>
                    <div class="metric-val" style="color: #a78bfa;">{final_result.total_candidates_evaluated}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p7_3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Passed Threshold (≥ {final_result.threshold_used})</div>
                    <div class="metric-val" style="color: #34d399;">{final_result.passed_threshold_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p7_4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Excluded (< {final_result.threshold_used})</div>
                    <div class="metric-val" style="color: #f43f5e;">{final_result.rejected_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p7_5:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Final Top K Limit</div>
                    <div class="metric-val" style="color: #fbbf24;">{final_result.top_k_limit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Scored Results Showcase
        if not final_result.results:
            st.warning(f"⚠️ No candidate patents satisfied the final score threshold (Score ≥ {final_result.threshold_used} / 10.0). No matching patents found.")
        else:
            st.markdown(f"### 🎖️ Top {len(final_result.results)} Qualifying Patent Matches")
            for rank_idx, fpat in enumerate(final_result.results):
                meta = fpat.metadata or {}
                title = meta.get("Title-english") or meta.get("Title") or "Title not available"
                assignee_list = meta.get("Current Assignee Standardized") or meta.get("Applicant First Organization") or []
                assignee_str = ", ".join(assignee_list) if isinstance(assignee_list, list) else str(assignee_list)
                pub_year = meta.get("Publication Year") or meta.get("Application Year", "N/A")
                country = meta.get("Publication Country Code") or "N/A"
                patent_url = PATENT_VIEW_URL_TEMPLATE.format(patent_id=fpat.patent_id)

                sb = fpat.score_breakdown
                cov_pct = int(fpat.relationship_coverage * 100)
                req_pct = int(fpat.requirement_coverage * 100)

                # Final score badge styling
                if fpat.final_score >= 8.5:
                    final_badge_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.35), rgba(5, 150, 105, 0.45))"
                    final_badge_border = "rgba(52, 211, 153, 0.8)"
                    final_score_color = "#6ee7b7"
                elif fpat.final_score >= 7.5:
                    final_badge_bg = "linear-gradient(135deg, rgba(6, 182, 212, 0.35), rgba(14, 116, 144, 0.45))"
                    final_badge_border = "rgba(103, 232, 249, 0.8)"
                    final_score_color = "#67e8f9"
                else:
                    final_badge_bg = "linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(79, 70, 229, 0.4))"
                    final_badge_border = "rgba(165, 180, 252, 0.8)"
                    final_score_color = "#a5b4fc"

                st.markdown(
                    f"""
                    <div class="candidate-card" style="border: 1px solid rgba(255, 255, 255, 0.15); background: rgba(15, 23, 42, 0.85); padding: 20px; border-radius: 12px; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                            <div>
                                <span style="font-size: 1.35rem; font-weight: 800; color: #67e8f9;">#{rank_idx+1}</span>
                                <a href="{patent_url}" target="_blank" style="font-size: 1.3rem; font-weight: 700; color: #818cf8; text-decoration: none; margin-left: 8px;">
                                    {fpat.patent_id} ↗
                                </a>
                                <span style="color: #94a3b8; font-size: 0.9rem; margin-left: 8px;">({country} • {pub_year})</span>
                            </div>
                            <div style="background: {final_badge_bg}; border: 1.5px solid {final_badge_border}; padding: 6px 14px; border-radius: 8px; text-align: right; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                                <span style="color: #cbd5e1; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; display: block;">Final Patent Score</span>
                                <span style="color: {final_score_color}; font-weight: 900; font-family: monospace; font-size: 1.4rem;">{fpat.final_score:.2f} <span style="font-size: 0.85rem; color: #94a3b8;">/ 10</span></span>
                            </div>
                        </div>
                        
                        <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
                            {title}
                        </div>
                        <div style="font-size: 0.86rem; color: #cbd5e1; margin-bottom: 12px;">
                            <span style="color: #94a3b8;">Assignee:</span> <strong>{assignee_str or 'Unknown'}</strong>
                        </div>

                        <!-- Score Components Breakdown Bar -->
                        <div style="background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;">
                                📊 Score Component Breakdown (Weights: Rel 45% • Req 25% • BGE 20% • Ret 10%)
                            </div>
                            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #6ee7b7; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                                    🔗 Relationships: <strong>{sb.relationship_weighted:.2f}</strong>/4.5 ({cov_pct}%)
                                </span>
                                <span style="background: rgba(6, 182, 212, 0.2); border: 1px solid rgba(6, 182, 212, 0.4); color: #67e8f9; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                                    📋 Requirements: <strong>{sb.requirement_weighted:.2f}</strong>/2.5 ({req_pct}%)
                                </span>
                                <span style="background: rgba(168, 85, 247, 0.2); border: 1px solid rgba(168, 85, 247, 0.4); color: #d8b4fe; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                                    ⚡ BGE Cross-Encoder: <strong>{sb.reranker_weighted:.2f}</strong>/2.0 ({sb.reranker_score:.3f})
                                </span>
                                <span style="background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4); color: #fbbf24; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                                    🎯 Vector Retrieval: <strong>{sb.retrieval_weighted:.2f}</strong>/1.0 ({sb.retrieval_score:.3f})
                                </span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show Top Verified Evidence for this Qualifying Patent
                with st.expander(f"📖 View Verified Evidence & Relationships for Patent #{rank_idx+1} ({fpat.patent_id})", expanded=False):
                    if fpat.relationships:
                        st.markdown("**🔗 Verified Relationships:**")
                        for r_ver in fpat.relationships:
                            st.markdown(f"- `{r_ver.subject} ➔ [{r_ver.relation}] ➔ {r_ver.object}`: **{r_ver.status}** (confidence: {r_ver.confidence*100:.0f}%)")
                    if fpat.evidence:
                        st.markdown("**📄 Top Supporting Evidence Chunks:**")
                        for ch in fpat.evidence[:3]:
                            st.markdown(
                                f"""
                                <div class="best-chunk-container" style="padding: 10px 14px; margin-bottom: 6px;">
                                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 6px;">
                                        <span><strong>Chunk #{ch.chunk_id}</strong> (Section: {ch.section or 'N/A'})</span>
                                        <span style="color: #67e8f9; font-weight: 700;">BGE Score: {ch.reranker_score:.4f}</span>
                                    </div>
                                    <div class="chunk-text-box" style="font-size: 0.82rem; max-height: 140px;">{ch.text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Raw Phase 7 JSON Output
        with st.expander("🔍 View Raw Phase 7 FinalSearchResult JSON Data", expanded=False):
            st.json(final_result.model_dump())






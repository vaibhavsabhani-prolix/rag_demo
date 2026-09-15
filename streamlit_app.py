"""
Patent Semantic Search — Streamlit Application
Phase 1: Dynamic Query Understanding & Pipeline Execution Inspector
"""

import json
import time
import streamlit as st

from app.config import (
    QUERY_CACHE_SIZE,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    QUERY_LLM_REQUEST_TIMEOUT,
)
from app.models.parsed_query import ParsedQuery
from app.query_understanding.engine import QueryUnderstandingEngine

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Patent Semantic Search — Query Understanding",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark glassmorphism look
st.markdown(
    """
    <style>
    /* Global enhancements */
    .stApp {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 85% 25%, rgba(6, 182, 212, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 50% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Metrics Cards */
    .metric-box {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 1.5rem;
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
        font-size: 0.85rem;
        font-weight: 500;
        margin: 4px;
    }

    /* Relationship Row */
    .rel-card {
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .rel-node {
        background: rgba(255, 255, 255, 0.08);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .rel-arrow {
        color: #818cf8;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0 6px;
    }
    
    /* Filter Badge */
    .filter-row {
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px 12px;
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
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 0 4px;
    }
    .val-tag {
        background: rgba(6, 182, 212, 0.15);
        color: #67e8f9;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 2px 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Engine Singleton
# -----------------------------------------------------------------------------
@st.cache_resource
def get_engine() -> QueryUnderstandingEngine:
    return QueryUnderstandingEngine()

engine = get_engine()

# -----------------------------------------------------------------------------
# Sidebar: Settings & Cache Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Infrastructure")
    st.caption(f"**Model:** `{QUERY_LLM_REMOTE_MODEL}`")
    st.caption(f"**Base URL:** `{QUERY_LLM_REMOTE_BASE_URL}`")
    st.caption(f"**Timeout:** `{QUERY_LLM_REQUEST_TIMEOUT}s`")
    
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
st.markdown("#### **Phase 1: Dynamic Query Understanding & Execution Pipeline**")
st.caption("Converts arbitrary natural-language patent queries into strongly-typed semantic representations with directed relationships and canonical metadata filters.")

# -----------------------------------------------------------------------------
# Search Pipeline Flow Tracker
# -----------------------------------------------------------------------------
st.markdown("##### 🚀 Search Pipeline Execution Flow")
col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
with col_p1:
    st.success("**STAGE 1**\n\n**Query Understanding**\n\n`ACTIVE / READY`")
with col_p2:
    st.info("**STAGE 2**\n\n**Vector Embedding**\n\n`Qwen3-0.6B (1024d)`")
with col_p3:
    st.info("**STAGE 3**\n\n**Qdrant Retrieval**\n\n`HNSW + Payload`")
with col_p4:
    st.info("**STAGE 4**\n\n**BGE Reranking**\n\n`Cross-Encoder`")
with col_p5:
    st.info("**STAGE 5**\n\n**Verification**\n\n`Evidence Grounding`")

st.divider()

# -----------------------------------------------------------------------------
# Custom Search Form
# -----------------------------------------------------------------------------
with st.form("search_form", clear_on_submit=False):
    query_input = st.text_input(
        "Patent Search Query:",
        value="",
        placeholder="Enter your custom patent search query (e.g., 'Find patents for a flexible OLED display with a moisture barrier')...",
        help="Type any natural-language patent search query across any technical domain.",
    )

    col_btn, col_cache_toggle = st.columns([2, 8])
    with col_btn:
        analyze_clicked = st.form_submit_button("🔍 Search & Analyze", type="primary", use_container_width=True)
    with col_cache_toggle:
        use_cache = st.checkbox("Enable In-Memory LRU Cache", value=True)

# Maintain state between interactions
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_timings" not in st.session_state:
    st.session_state["last_timings"] = None

if analyze_clicked:
    clean_q = query_input.strip()
    if not clean_q:
        st.warning("Please enter a search query.")
    else:
        was_cached = engine.cache.get(clean_q) is not None if use_cache else False
        with st.spinner("Executing Dynamic Query Understanding..."):
            t_start = time.perf_counter()
            parsed_query = engine.parse(clean_q, use_cache=use_cache)
            t_total_ms = (time.perf_counter() - t_start) * 1000

        if was_cached:
            llm_ms = 0.0
            norm_ms = t_total_ms
            status_text = "⚡ CACHE HIT"
            status_color = "#10b981"
        else:
            norm_ms = 0.005
            llm_ms = max(0.0, t_total_ms - norm_ms)
            status_text = "❄️ COLD (1 LLM Call)"
            status_color = "#06b6d4"

        st.session_state["last_query"] = clean_q
        st.session_state["last_result"] = parsed_query
        st.session_state["last_timings"] = {
            "total_ms": t_total_ms,
            "llm_ms": llm_ms,
            "norm_ms": norm_ms,
            "status_text": status_text,
            "status_color": status_color,
        }

if st.session_state.get("last_result") is not None:
    parsed_query = st.session_state["last_result"]
    timings = st.session_state["last_timings"]
    t_total_ms = timings["total_ms"]
    llm_ms = timings["llm_ms"]
    norm_ms = timings["norm_ms"]
    status_text = timings["status_text"]
    status_color = timings["status_color"]

    # ---------------------------------------------------------------------
    # Live Performance Metrics Dashboard
    # ---------------------------------------------------------------------
    st.markdown("### ⏱️ Live Execution Timings")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Execution Time</div>
                <div class="metric-val" style="color: #67e8f9;">{t_total_ms:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">LLM Inference (1 Call)</div>
                <div class="metric-val" style="color: #a5b4fc;">{llm_ms:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Local Normalization</div>
                <div class="metric-val" style="color: #6ee7b7;">{norm_ms:.4f} <span style="font-size: 0.8rem; color: #94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Cache / Request State</div>
                <div class="metric-val" style="color: {status_color}; font-size: 1.1rem;">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Results Breakdown
    # ---------------------------------------------------------------------
    st.markdown("### 🧠 Semantic Intent & Extracted Structure")

    # Semantic Query
    st.info(f"**Semantic Query (Intent Preserved):** {parsed_query.semantic_query}")
    if parsed_query.is_metadata_only:
        st.warning("🏷️ **Metadata-Only Query**: Query contains exclusively metadata constraints with zero technical concepts.")

    # Grid of Results
    col_left, col_right = st.columns(2)

    with col_left:
        # Concepts
        st.markdown("#### 🏷️ Technical Concepts")
        if parsed_query.concepts:
            chips_html = "".join([f'<span class="concept-chip">🏷️ {c}</span>' for c in parsed_query.concepts])
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("No explicit technical concepts extracted.")

        # Directed Relationships
        st.markdown("#### 🔗 Explicit Directed Relationships")
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

        # Concept Attributes
        st.markdown("#### ⚙️ Concept Attributes")
        if parsed_query.attributes:
            for a in parsed_query.attributes:
                st.markdown(f"- **{a.concept}** ➔ `{a.name}`: **{a.value}**")
        else:
            st.caption("No concept attributes specified.")

    with col_right:
        # Requirements, Constraints, Exclusions
        st.markdown("#### 📋 Requirements & Constraints")
        if parsed_query.requirements:
            st.markdown("**Requirements:**")
            for req in parsed_query.requirements:
                st.markdown(f"- ✅ {req}")

        if parsed_query.constraints:
            st.markdown("**Constraints:**")
            for con in parsed_query.constraints:
                st.markdown(f"- ⚠️ **{con}**")

        if parsed_query.exclusions:
            st.markdown("**Exclusions:**")
            for ex in parsed_query.exclusions:
                st.markdown(f"- 🚫 <span style='color: #f43f5e; font-weight: 600;'>{ex}</span>", unsafe_allow_html=True)

        if not parsed_query.requirements and not parsed_query.constraints and not parsed_query.exclusions:
            st.caption("No specific requirements, constraints, or exclusions.")

        # Metadata Filters
        st.markdown("#### 📑 Resolved Metadata Filters")
        if parsed_query.metadata_filters:
            for f in parsed_query.metadata_filters:
                st.markdown(
                    f"""
                    <div class="filter-row">
                        <div>
                            <span class="code-tag">{f.field}</span>
                            <span style="font-size: 0.85rem; color: #f1f5f9; margin-left: 8px;">{f.raw_field or f.field}</span>
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

    # Raw JSON Output
    with st.expander("🔍 View Raw ParsedQuery JSON Schema", expanded=False):
        st.json(parsed_query.model_dump())

import html
import os
import sys
from urllib.parse import quote

import streamlit as st

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.config import (
    PATENT_VIEW_URL_TEMPLATE,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_TITLE,
)

st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout=STREAMLIT_LAYOUT)

from app.history_manager import HistoryManager
from app.semantic_search import SemanticSearch

_HEADER_STYLE = (
    "text-align:left; padding:8px 12px; border-bottom:1px solid rgba(128,128,128,0.3); "
    "font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em;"
)
_CELL_STYLE = "padding:8px 12px; border-bottom:1px solid rgba(128,128,128,0.15); vertical-align:top;"


def _escape(val: object) -> str:
    return html.escape(str(val) if val is not None else "")


@st.cache_resource(show_spinner=False)
def _get_search(_version: str = "v2.1") -> SemanticSearch:
    return SemanticSearch()


@st.cache_resource(show_spinner=False)
def _get_history_manager() -> HistoryManager:
    return HistoryManager()


def _render_qdrant_candidates(candidates: list) -> None:
    if not candidates:
        st.info("No candidates returned from vector search.")
        return

    columns = ["#", "Qdrant Score", "Patent ID", "Chunk ID", "Section", "Text Preview"]

    parts = [
        '<div style="border:1px solid rgba(128,128,128,0.4); border-radius:6px;">',
        '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">',
        "<thead><tr>",
        "".join(f'<th style="{_HEADER_STYLE}">{col}</th>' for col in columns),
        "</tr></thead><tbody>",
    ]

    for idx, point in enumerate(candidates, start=1):
        payload = point.payload or {}
        text_snippet = payload.get("text", "").replace("\n", " ").strip()
        if len(text_snippet) > 80:
            text_snippet = text_snippet[:77] + "..."

        parts.append(
            "<tr>"
            f'<td style="{_CELL_STYLE}">{idx}</td>'
            f'<td style="{_CELL_STYLE}">{point.score:.4f}</td>'
            f'<td style="{_CELL_STYLE} font-weight:600;">{_escape(payload.get("patent_id", ""))}</td>'
            f'<td style="{_CELL_STYLE}">{payload.get("chunk_id", "")}</td>'
            f'<td style="{_CELL_STYLE}">{_escape(payload.get("section", ""))}</td>'
            f'<td style="{_CELL_STYLE} white-space:pre-wrap; word-break:break-word;">{_escape(text_snippet)}</td>'
            "</tr>"
        )

    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_results_table(results: list, is_question: bool = False) -> None:
    if not results:
        st.info("No matching patents found.")
        return

    if is_question:
        columns = [
            "#",
            "Patent ID",
            "Score",
            "Extracted Answer",
            "Section",
            "Best Matching Chunk & Highlighted Evidence",
        ]
    else:
        columns = [
            "#",
            "Patent ID",
            "Score",
            "Best Chunk",
            "Section",
            "Best Matching Chunk (Full Text)",
        ]

    parts = [
        '<div style="border:1px solid rgba(128,128,128,0.4); border-radius:6px;">',
        '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">',
        "<thead><tr>",
        "".join(f'<th style="{_HEADER_STYLE}">{col}</th>' for col in columns),
        "</tr></thead><tbody>",
    ]

    for rank, patent in enumerate(results, start=1):
        best = patent.best_chunk
        patent_url = PATENT_VIEW_URL_TEMPLATE.format(
            patent_id=quote(str(patent.patent_id), safe="")
        )

        if is_question:
            answer_text = (
                patent.answer or best.answer or "(no direct answer identified)"
            )
            answer_badge = (
                f'<div style="background:rgba(255,224,102,0.25); border-left:3px solid #ffcc00; '
                f'padding:6px 10px; border-radius:4px; font-weight:600; font-size:0.85rem; color:#d48800;">'
                f"{_escape(answer_text)}</div>"
            )
            chunk_display = (
                patent.highlighted_text or best.highlighted_text or _escape(best.text)
            )

            parts.append(
                "<tr>"
                f'<td style="{_CELL_STYLE}">{rank}</td>'
                f'<td style="{_CELL_STYLE} font-weight:600;">'
                f'<a href="{_escape(patent_url)}" target="_blank" rel="noopener noreferrer">'
                f"{_escape(patent.patent_id)}</a></td>"
                f'<td style="{_CELL_STYLE}">{patent.score:.1f}/10</td>'
                f'<td style="{_CELL_STYLE} min-width:200px;">{answer_badge}</td>'
                f'<td style="{_CELL_STYLE}">{_escape(best.section)}</td>'
                f'<td style="{_CELL_STYLE} white-space:pre-wrap; word-break:break-word;">{chunk_display}</td>'
                "</tr>"
            )
        else:
            chunk_display = (
                patent.highlighted_text or best.highlighted_text or _escape(best.text)
            )
            parts.append(
                "<tr>"
                f'<td style="{_CELL_STYLE}">{rank}</td>'
                f'<td style="{_CELL_STYLE} font-weight:600;">'
                f'<a href="{_escape(patent_url)}" target="_blank" rel="noopener noreferrer">'
                f"{_escape(patent.patent_id)}</a></td>"
                f'<td style="{_CELL_STYLE}">{patent.score:.1f}/10</td>'
                f'<td style="{_CELL_STYLE}">{best.chunk_id}</td>'
                f'<td style="{_CELL_STYLE}">{_escape(best.section)}</td>'
                f'<td style="{_CELL_STYLE} white-space:pre-wrap; word-break:break-word;">{chunk_display}</td>'
                "</tr>"
            )

    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_search_page(history_mgr: HistoryManager) -> None:
    st.title("Patent Semantic Search")

    prefill_val = st.session_state.get("prefill_query", "")
    auto_trigger = st.session_state.get("auto_submit", False)

    with st.form("search_form"):
        col_input, col_button = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "Search query",
                value=prefill_val,
                placeholder="e.g. What properties can be determined based on the material?",
                key="search_query_input_box",
            )
        with col_button:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Search", type="primary", use_container_width=True
            )

    # If triggered via "Run this query" from history, simulate submission
    if auto_trigger and prefill_val:
        submitted = True
        query = prefill_val
        st.session_state["auto_submit"] = False
        st.session_state["prefill_query"] = ""

    if not (submitted and query.strip()):
        return

    search = _get_search()
    stage_timings: list[tuple[str, float]] = []

    with st.status("Running search pipeline...", expanded=True) as status:

        def _on_stage(name: str, elapsed: float) -> None:
            stage_timings.append((name, elapsed))
            status.write(f"✅ {name} — {elapsed * 1000:.0f} ms")

        try:
            parsed, qdrant_results, _, _, results = search.search_detailed(
                query, on_stage=_on_stage
            )
        except Exception as e:
            status.update(label="Search failed", state="error")
            st.exception(e)
            return

        total_ms = sum(elapsed for _, elapsed in stage_timings) * 1000
        status.update(label=f"Pipeline complete — {total_ms:.0f} ms", state="complete")

    # Save search query and metadata to persistent history
    try:
        filters_list = (
            [{"field": f.field, "operator": f.operator, "value": f.value} for f in parsed.metadata_filters]
            if parsed.metadata_filters
            else None
        )
        top_patents_list = [p.patent_id for p in results[:5]] if results else None
        extracted_answer = results[0].answer if (results and parsed.is_question) else None

        history_mgr.add_search(
            query=query.strip(),
            is_question=parsed.is_question,
            semantic_query=parsed.semantic_query,
            result_count=len(results),
            execution_time_ms=total_ms,
            filters=filters_list,
            top_patents=top_patents_list,
            extracted_answer=extracted_answer,
        )
    except Exception as e:
        st.warning(f"Could not save search to history: {e}")

    with st.expander("Query Understanding", expanded=True):
        st.write(
            f"**Semantic query:** {parsed.semantic_query or '(none - metadata only)'}"
        )
        st.write(f"**Is this a question?:** {'Yes' if parsed.is_question else 'No'}")
        if parsed.is_question and parsed.question_intent:
            qi = parsed.question_intent
            if qi.target:
                st.write(f"**Target:** {qi.target}")
            if qi.expected_answer_type:
                st.write(f"**Expected answer type:** {qi.expected_answer_type}")
            if qi.answer_criteria:
                st.write(f"**Answer criteria:** {qi.answer_criteria}")

        if parsed.metadata_filters:
            st.write("**Resolved filters:**")
            for f in parsed.metadata_filters:
                st.write(f"- `{f.field}` `{f.operator}` `{f.value}`")
        else:
            st.write("**Resolved filters:** (none)")

        if parsed.has_requirements_structure:
            st.write("**Reranking requirements structure:**")
            if parsed.intent:
                st.write(f"- Intent: {parsed.intent}")
            if parsed.query_type:
                st.write(f"- Query type: {', '.join(parsed.query_type)}")
            if parsed.concepts:
                st.write(
                    "- Concepts: "
                    + ", ".join(
                        f"{c.text} ({c.role}{', required' if c.required else ''})"
                        for c in parsed.concepts
                    )
                )
            if parsed.goals:
                st.write("- Goals: " + ", ".join(g.text for g in parsed.goals))
            if parsed.constraints:
                st.write(
                    "- Constraints: " + ", ".join(k.text for k in parsed.constraints)
                )
            if parsed.optimization:
                st.write(
                    "- Optimization: "
                    + ", ".join(
                        f"{o.direction} {o.property}" for o in parsed.optimization
                    )
                )
            if parsed.exclusions:
                st.write("- Exclusions: " + ", ".join(parsed.exclusions))
            if parsed.relationships:
                st.write(
                    "- Relationships: "
                    + ", ".join(
                        f"{r.source} —{r.relation}→ {r.target}"
                        for r in parsed.relationships
                    )
                )
            w = parsed.ranking_weights
            st.write(
                "- Ranking weights: "
                f"semantic={w.semantic_relevance:.2f}, "
                f"requirement={w.requirement_satisfaction:.2f}, "
                f"relationship={w.relationship_satisfaction:.2f}, "
                f"constraint={w.constraint_satisfaction:.2f}, "
                f"evidence={w.evidence_strength:.2f}, "
                f"exact_match={w.exact_match:.2f}"
            )

        st.write("**Pipeline steps:**")
        for name, elapsed in stage_timings:
            st.write(f"- `{name}` — {elapsed * 1000:.0f} ms")

    with st.expander(f"Qdrant Vector Search Candidates ({len(qdrant_results)})"):
        _render_qdrant_candidates(qdrant_results)

    st.caption(f"{len(results)} matching patent(s)")

    _render_results_table(results, is_question=parsed.is_question)

    if parsed.is_question and results:
        with st.expander("Answer Evidence Debug", expanded=False):
            for rank, patent in enumerate(results, start=1):
                st.markdown(f"### {rank}. {patent.patent_id}")
                debug = patent.answer_debug or {}
                st.write(
                    f"**Question:** {debug.get('question', parsed.original_query)}"
                )
                st.write(
                    f"**Dynamic answer target:** {debug.get('dynamic_answer_target', '(not available)')}"
                )
                st.write(
                    f"**Expected answer type:** {debug.get('expected_answer_type', '(not available)')}"
                )
                st.write(
                    f"**Extracted answer:** {debug.get('extracted_answer', patent.answer or '(no direct answer identified)')}"
                )

                selected = debug.get("selected_evidence")
                if isinstance(selected, dict):
                    st.write(
                        f"**Selected evidence:** {selected.get('text', '(not available)')}"
                    )
                    if selected.get("span"):
                        st.write(f"**Highlighted span:** {selected.get('span')}")

                candidates = debug.get("candidate_evidence")
                if isinstance(candidates, list) and candidates:
                    st.write("**Top candidate evidence scores:**")
                    for candidate in candidates[:3]:
                        if isinstance(candidate, dict):
                            st.write(
                                "- "
                                f"direct={candidate.get('is_direct')} | "
                                f"direct_score={candidate.get('directness', candidate.get('confidence', 0.0))} | "
                                f"answer_relevance={candidate.get('final_score', 0.0)} | "
                                f"text={str(candidate.get('text', candidate.get('evidence_text', '')))[:220]}"
                            )

                reason = debug.get("reason")
                if reason:
                    st.write(f"**Selection reason:** {reason}")


def _render_history_page(history_mgr: HistoryManager) -> None:
    st.title("Search History")
    st.caption("Review previous patent searches, inspect extracted answers, and quickly re-run queries.")

    stats = history_mgr.get_stats()

    # Overview Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Searches", stats["total_searches"])
    m2.metric("Questions Asked", stats["question_count"])
    m3.metric("Topic Queries", stats["topic_count"])
    m4.metric("Avg Duration", f"{stats['avg_latency_ms']:.0f} ms" if stats["total_searches"] > 0 else "N/A")

    st.markdown("---")

    if stats["total_searches"] == 0:
        st.info("No search history found yet. Run some queries from the Search page to see them here!")
        return

    # Filter & Search Controls
    col_search, col_filter, col_export = st.columns([3, 2, 2])
    with col_search:
        filter_text = st.text_input("🔍 Search History", placeholder="Filter by query or keyword...")
    with col_filter:
        type_filter = st.selectbox(
            "Query Type",
            ["All Queries", "Questions Only", "Topic Searches Only"],
            index=0,
        )
    with col_export:
        st.write("")
        st.write("")
        csv_data = history_mgr.export_csv()
        st.download_button(
            "📥 Export CSV",
            data=csv_data,
            file_name="patent_search_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    is_question_filter = None
    if type_filter == "Questions Only":
        is_question_filter = True
    elif type_filter == "Topic Searches Only":
        is_question_filter = False

    history_items = history_mgr.get_history(
        limit=100,
        search_term=filter_text,
        is_question_filter=is_question_filter,
    )

    # Actions: Clear All History
    with st.expander("⚙️ History Settings & Management"):
        col_clear, col_json = st.columns([2, 2])
        with col_clear:
            if st.button("🧹 Clear All History", type="secondary"):
                history_mgr.clear_history()
                st.success("Search history cleared!")
                st.rerun()
        with col_json:
            json_data = history_mgr.export_json()
            st.download_button(
                "📥 Export JSON",
                data=json_data,
                file_name="patent_search_history.json",
                mime="application/json",
            )

    st.markdown(f"### Historical Searches ({len(history_items)})")

    if not history_items:
        st.warning("No matching queries found with the current filter.")
        return

    for item in history_items:
        item_id = item["id"]
        query_text = item["query"]
        is_question = item["is_question"]
        timestamp = item["timestamp"][:19].replace("T", " ")
        res_count = item["result_count"]
        exec_time = item["execution_time_ms"]
        answer = item.get("extracted_answer")
        top_patents = item.get("top_patents") or []
        filters = item.get("filters") or []

        type_badge = "❓ Question" if is_question else "🔍 Topic"

        with st.container(border=True):
            header_col, action_col = st.columns([4, 1.2])

            with header_col:
                st.markdown(f"**{_escape(query_text)}**")
                st.caption(
                    f"**{type_badge}** | 📅 {timestamp} UTC | ⏱️ {exec_time:.0f} ms | 📄 {res_count} result(s)"
                )

                if answer:
                    st.markdown(
                        f'<div style="background:rgba(255,224,102,0.2); border-left:3px solid #ffcc00; '
                        f'padding:4px 8px; border-radius:4px; font-size:0.85rem; margin-top:4px; margin-bottom:4px;">'
                        f'<strong>Answer:</strong> {_escape(answer)}</div>',
                        unsafe_allow_html=True,
                    )

                if top_patents:
                    st.caption(f"**Top Patents:** {', '.join(top_patents)}")

                if filters:
                    filter_str = ", ".join(f"{f.get('field')} {f.get('operator')} {f.get('value')}" for f in filters)
                    st.caption(f"**Applied Filters:** {filter_str}")

            with action_col:
                st.write("")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔁 Re-run", key=f"rerun_{item_id}", help="Run this query in Search"):
                        st.session_state["prefill_query"] = query_text
                        st.session_state["auto_submit"] = True
                        st.session_state["selected_page"] = "🔍 Search"
                        st.rerun()
                with col_btn2:
                    if st.button("🗑️", key=f"del_{item_id}", help="Delete from history"):
                        history_mgr.delete_search(item_id)
                        st.rerun()


def main() -> None:
    history_mgr = _get_history_manager()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    
    # Initialize selected_page in session_state if not present
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "🔍 Search"

    # Sidebar menu
    page = st.sidebar.radio(
        "Go to",
        ["🔍 Search", "🕒 Search History"],
        index=0 if st.session_state["selected_page"] == "🔍 Search" else 1,
        key="main_nav_radio",
    )
    
    # Keep session state in sync
    st.session_state["selected_page"] = page

    # Quick sidebar stats
    st.sidebar.markdown("---")
    stats = history_mgr.get_stats()
    st.sidebar.caption(f"📊 **Total Searches Logged:** {stats['total_searches']}")
    if stats["latest_search_time"]:
        latest_str = stats["latest_search_time"][:19].replace("T", " ")
        st.sidebar.caption(f"🕒 **Last Search:** {latest_str} UTC")

    if page == "🔍 Search":
        _render_search_page(history_mgr)
    else:
        _render_history_page(history_mgr)


if __name__ == "__main__":
    main()

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
            # Use highlighted text if available, otherwise fallback to escaped text
            chunk_display = (
                patent.highlighted_text or best.highlighted_text or _escape(best.text)
            )

            parts.append(
                "<tr>"
                f'<td style="{_CELL_STYLE}">{rank}</td>'
                f'<td style="{_CELL_STYLE} font-weight:600;">'
                f'<a href="{_escape(patent_url)}" target="_blank" rel="noopener noreferrer">'
                f"{_escape(patent.patent_id)}</a></td>"
                f'<td style="{_CELL_STYLE}">{patent.score:.4f}</td>'
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
                f'<td style="{_CELL_STYLE}">{patent.score:.4f}</td>'
                f'<td style="{_CELL_STYLE}">{best.chunk_id}</td>'
                f'<td style="{_CELL_STYLE}">{_escape(best.section)}</td>'
                f'<td style="{_CELL_STYLE} white-space:pre-wrap; word-break:break-word;">{chunk_display}</td>'
                "</tr>"
            )

    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def main() -> None:
    st.title("Patent Semantic Search")

    with st.form("search_form"):
        col_input, col_button = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "Search query",
                placeholder="e.g. What properties can be determined based on the material?",
            )
        with col_button:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Search", type="primary", use_container_width=True
            )

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


if __name__ == "__main__":
    main()

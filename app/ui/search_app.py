"""
Search UI

A Streamlit front-end for the search pipeline ONLY - a search bar and a
results table. No ingestion, indexing, or admin actions live here; this
page just drives SemanticSearch.search_detailed(), the same pipeline
app/_tests_/test_semantic_search.py exercises from the CLI.

One row per patent. Unlike the CLI's truncated preview, the "Best
Matching Chunk" cell shows that chunk's FULL text, un-truncated - the
patent's other matching chunks are not shown here at all.

Run:
    streamlit run app/ui/search_app.py
"""

import html
import sys
from pathlib import Path
from urllib.parse import quote

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.config import PATENT_VIEW_URL_TEMPLATE
from app.semantic_search import SemanticSearch

st.set_page_config(page_title="Patent Semantic Search", layout="wide")

st.markdown(
    "<style>.block-container { padding-top: 8px; }</style>",
    unsafe_allow_html=True,
)

_HEADER_STYLE = (
    "text-align:left; padding:6px 10px; position:sticky; top:0; "
    "background:rgba(128,128,128,0.18); border-bottom:1px solid rgba(128,128,128,0.4);"
)
_CELL_STYLE = "padding:6px 10px; border-bottom:1px solid rgba(128,128,128,0.25); vertical-align:top;"


@st.cache_resource(show_spinner="Loading search pipeline (embedder, reranker, query understanding)...")
def _get_search() -> SemanticSearch:
    return SemanticSearch()


def _escape(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _render_qdrant_candidates(qdrant_results: list) -> None:
    if not qdrant_results:
        st.write("(no candidates)")
        return

    columns = ["#", "Patent ID", "Score", "Chunk ID", "Section", "Best Matching Chunk"]

    parts = [
        '<div style="border:1px solid rgba(128,128,128,0.4); border-radius:6px;">',
        '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">',
        "<thead><tr>",
        "".join(f'<th style="{_HEADER_STYLE}">{col}</th>' for col in columns),
        "</tr></thead><tbody>",
    ]

    ranked = sorted(qdrant_results, key=lambda p: p.score, reverse=True)

    for rank, point in enumerate(ranked, start=1):
        payload = point.payload or {}
        parts.append(
            "<tr>"
            f'<td style="{_CELL_STYLE}">{rank}</td>'
            f'<td style="{_CELL_STYLE} font-weight:600;">{_escape(payload.get("patent_id"))}</td>'
            f'<td style="{_CELL_STYLE}">{point.score:.4f}</td>'
            f'<td style="{_CELL_STYLE}">{_escape(payload.get("chunk_id"))}</td>'
            f'<td style="{_CELL_STYLE}">{_escape(payload.get("section"))}</td>'
            f'<td style="{_CELL_STYLE} white-space:pre-wrap; word-break:break-word;">{_escape(payload.get("text"))}</td>'
            "</tr>"
        )

    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_results_table(results: list) -> None:
    if not results:
        st.info("No matching patents found.")
        return

    columns = ["#", "Patent ID", "Score", "Best Chunk", "Section", "Best Matching Chunk (Full Text)"]

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
        parts.append(
            "<tr>"
            f'<td style="{_CELL_STYLE}">{rank}</td>'
            f'<td style="{_CELL_STYLE} font-weight:600;">'
            f'<a href="{_escape(patent_url)}" target="_blank" rel="noopener noreferrer">'
            f'{_escape(patent.patent_id)}</a></td>'
            f'<td style="{_CELL_STYLE}">{patent.score:.4f}</td>'
            f'<td style="{_CELL_STYLE}">{best.chunk_id}</td>'
            f'<td style="{_CELL_STYLE}">{_escape(best.section)}</td>'
            f'<td style="{_CELL_STYLE} white-space:pre-wrap; word-break:break-word;">{_escape(best.text)}</td>'
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
                placeholder="e.g. biodegradable polymer composition filed in AP in 2007",
            )
        with col_button:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    if not (submitted and query.strip()):
        return

    search = _get_search()

    stage_timings: list[tuple[str, float]] = []

    with st.status("Running search pipeline...", expanded=True) as status:

        def _on_stage(name: str, elapsed: float) -> None:
            stage_timings.append((name, elapsed))
            status.write(f"✅ {name} — {elapsed * 1000:.0f} ms")

        try:
            parsed, qdrant_results, _, _, results = search.search_detailed(query, on_stage=_on_stage)
        except Exception as e:
            status.update(label="Search failed", state="error")
            st.exception(e)
            return

        total_ms = sum(elapsed for _, elapsed in stage_timings) * 1000
        status.update(label=f"Pipeline complete — {total_ms:.0f} ms", state="complete")

    with st.expander("Query Understanding", expanded=True):
        st.write(f"**Semantic query:** {parsed.semantic_query or '(none - metadata only)'}")
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
                st.write("- Constraints: " + ", ".join(k.text for k in parsed.constraints))
            if parsed.optimization:
                st.write(
                    "- Optimization: "
                    + ", ".join(f"{o.direction} {o.property}" for o in parsed.optimization)
                )
            if parsed.exclusions:
                st.write("- Exclusions: " + ", ".join(parsed.exclusions))
            if parsed.relationships:
                st.write(
                    "- Relationships: "
                    + ", ".join(
                        f"{r.source} —{r.relation}→ {r.target}" for r in parsed.relationships
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

    _render_results_table(results)


if __name__ == "__main__":
    main()

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.semantic_search import SemanticSearch

st.set_page_config(page_title="Patent Semantic Search", layout="wide")

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


def _render_results_table(results: list) -> None:
    if not results:
        st.info("No matching patents found.")
        return

    columns = ["#", "Patent ID", "Score", "Best Chunk", "Section", "Best Matching Chunk (Full Text)"]

    parts = [
        '<div style="max-height:75vh; overflow-y:auto; border:1px solid rgba(128,128,128,0.4); border-radius:6px;">',
        '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">',
        "<thead><tr>",
        "".join(f'<th style="{_HEADER_STYLE}">{col}</th>' for col in columns),
        "</tr></thead><tbody>",
    ]

    for rank, patent in enumerate(results, start=1):
        best = patent.best_chunk
        parts.append(
            "<tr>"
            f'<td style="{_CELL_STYLE}">{rank}</td>'
            f'<td style="{_CELL_STYLE} font-weight:600;">{_escape(patent.patent_id)}</td>'
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
        query = st.text_input(
            "Search query",
            placeholder="e.g. biodegradable polymer composition filed in AP in 2007",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if not (submitted and query.strip()):
        return

    search = _get_search()

    with st.spinner("Searching..."):
        try:
            parsed, _, _, _, results = search.search_detailed(query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    with st.expander("Query Understanding"):
        st.write(f"**Semantic query:** {parsed.semantic_query or '(none - metadata only)'}")
        if parsed.metadata_filters:
            st.write("**Resolved filters:**")
            for f in parsed.metadata_filters:
                st.write(f"- `{f.field}` `{f.operator}` `{f.value}`")
        else:
            st.write("**Resolved filters:** (none)")

    st.caption(f"{len(results)} matching patent(s)")

    _render_results_table(results)


if __name__ == "__main__":
    main()

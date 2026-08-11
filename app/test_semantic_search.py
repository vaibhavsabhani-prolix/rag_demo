"""
End-to-end Patent Semantic Search Test & Diagnostics

Displays the full pipeline at each stage:

    Original Query
    Semantic Query
    Candidate Filters
    Resolved Metadata Filters
    Initial Vector Candidates
    Unique Candidate Patents
    Metadata Matching Patents
    Candidates After Metadata Filtering
    Reranked Candidates
    Final Patents

Run with:
    ./.venv/bin/python -m app.test_semantic_search "your query here"
"""

import sys
import threading
import time
from app.query_understanding import ParsedQuery
from app.semantic_search import SemanticSearch


# ==============================================================
# Terminal Spinner (Loading Indicator)
# ==============================================================

class Spinner:
    """
    Animated terminal spinner that runs in a background thread.

    Usage:
        with Spinner("Searching"):
            # ... long operation ...
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Searching"):
        self._message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            frame = self._FRAMES[idx % len(self._FRAMES)]
            sys.stdout.write(f"\r\033[36m{frame}\033[0m {self._message}...")
            sys.stdout.flush()
            idx += 1
            self._stop_event.wait(0.08)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._message) + 10) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        if self._thread:
            self._thread.join()


def print_ascii_table(title: str, headers: list[str], rows: list[list[str]]):
    """
    Render a clean ASCII table with title, column headers, and rows.
    """
    print()
    print("-" * 100)
    print(f" {title.upper()} ")
    print("-" * 100)

    if not rows:
        print("No results found.")
        print("-" * 100)
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    divider = "-+-".join(["-" * w for w in col_widths])

    print("+" + "-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 2) + "+")
    print("| " + row_fmt.format(*headers) + " |")
    print("+" + divider + "+")
    for r in rows:
        print("| " + row_fmt.format(*r) + " |")
    print("+" + "-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 2) + "+")


def print_query_understanding(parsed: ParsedQuery):
    print()
    print("=" * 100)
    print(" QUERY UNDERSTANDING")
    print("=" * 100)
    print()
    print("Original Query:")
    print(f"  {parsed.original_query}")
    print()
    print("Semantic Query:")
    print(f"  {parsed.semantic_query}")
    print()
    print("Candidate Filters:")
    if not parsed.candidate_filters:
        print("  (none)")
    else:
        for cf in parsed.candidate_filters:
            print(f'  meaning="{cf.meaning}", value="{cf.value}"')
    print()
    print("Resolved Metadata Filters:")
    if not parsed.metadata_filters:
        print("  (none)")
    else:
        for f in parsed.metadata_filters:
            print(f"  {f.field} {f.operator} {f.value}")
    print()
    print("=" * 100)


def print_candidate_diagnostics(parsed: ParsedQuery, qdrant_results, filtered_results, reranked_results, results=None):
    print()
    print("-" * 60)
    print("PIPELINE DIAGNOSTICS")
    print("-" * 60)

    unique_before = len({p.payload.get("patent_id") for p in qdrant_results if p.payload})
    unique_after = len({p.payload.get("patent_id") for p in filtered_results if p.payload})
    final_patents = len(results) if results is not None else unique_after

    print(f"Initial Vector Candidates          : {len(qdrant_results)}")
    print(f"Unique Candidate Patents           : {unique_before}")

    if parsed.metadata_filters:
        print(f"Metadata Matching Patents          : {unique_after}")
    else:
        print("Metadata Matching Patents          : N/A (no filters applied)")

    print(f"Candidates After Metadata Filtering: {len(filtered_results)}")
    print(f"Reranked Candidates                : {len(reranked_results)}")
    print(f"Final Patents                      : {final_patents}")
    print("-" * 60)


def run_search(search: SemanticSearch, query: str):
    # Show spinner while the search pipeline runs
    with Spinner("Searching"):
        parsed, qdrant_results, filtered_results, reranked_results, results = search.search_detailed(query)

    print()
    print("=" * 100)
    print("Query :", query)
    print("=" * 100)

    print_query_understanding(parsed)
    print_candidate_diagnostics(parsed, qdrant_results, filtered_results, reranked_results, results)


    # -------------------------------------------------------------
    # Table 1: Qdrant DB Candidate Results (Vector Search)
    #
    # Skipped for metadata-only queries (no real topic - see
    # ParsedQuery.is_metadata_only) - there was no vector search, so
    # qdrant_results holds the whole-collection filter matches instead
    # and would be misleading labeled as "vector search results".
    # -------------------------------------------------------------
    qdrant_rank_map = {}

    if not parsed.is_metadata_only:
        qdrant_headers = ["#", "Qdrant Score", "Patent ID", "Chunk ID", "Section", "Text Preview"]
        qdrant_rows = []

        for idx, point in enumerate(qdrant_results, start=1):
            qdrant_rank_map[point.id] = idx
            payload = point.payload or {}
            text_snippet = payload.get("text", "").replace("\n", " ").strip()
            if len(text_snippet) > 45:
                text_snippet = text_snippet[:42] + "..."

            qdrant_rows.append([
                str(idx),
                f"{point.score:.4f}",
                str(payload.get("patent_id", "")),
                str(payload.get("chunk_id", "")),
                str(payload.get("section", "")),
                text_snippet,
            ])

        print_ascii_table("Table 1: Vector Search Results (Qdrant DB)", qdrant_headers, qdrant_rows)

    # -------------------------------------------------------------
    # Table 2: Reranked Results (Cross-Encoder Reranking)
    # -------------------------------------------------------------
    rerank_headers = ["#", "Rerank Score", "Qdrant Rank", "Patent ID", "Chunk ID", "Section", "Text Preview"]
    rerank_rows = []

    for idx, (score, point) in enumerate(reranked_results, start=1):
        payload = point.payload or {}
        orig_rank = qdrant_rank_map.get(point.id, "N/A")
        qdrant_rank_str = f"#{orig_rank}" if isinstance(orig_rank, int) else str(orig_rank)

        text_snippet = payload.get("text", "").replace("\n", " ").strip()
        if len(text_snippet) > 45:
            text_snippet = text_snippet[:42] + "..."

        rerank_rows.append([
            str(idx),
            f"{score:.4f}",
            qdrant_rank_str,
            str(payload.get("patent_id", "")),
            str(payload.get("chunk_id", "")),
            str(payload.get("section", "")),
            text_snippet,
        ])

    print_ascii_table("Table 2: Final Reranked Results (Cross-Encoder Reranker)", rerank_headers, rerank_rows)

    if not results:
        print("\nNo matching patents found.")
        print("=" * 100)
        return

    print("\n" + "=" * 100)
    print(" PATENT-LEVEL AGGREGATED DETAILS ")
    print("=" * 100)

    for index, patent in enumerate(results, start=1):

        print()
        print(f"Result {index}")
        print("-" * 40)

        print(f"Patent              : {patent.patent_id}")
        print(f"Overall Patent Score: {patent.score:.4f}")
        print(f"Matching Chunks     : {patent.chunk_count}")
        print(f"Sections            : {', '.join(patent.sections)}")

        print()
        print(f"Best Matching Chunk : {patent.best_chunk.chunk_id}")

        # Show additional matching chunks (if any)
        additional = [c for c in patent.matching_chunks if c is not patent.best_chunk]
        if additional:
            print()
            print("Additional Matching Chunks:")
            for chunk in additional:
                print(f"  Chunk {chunk.chunk_id:4d}  "
                      f"Section: {chunk.section:30s}  "
                      f"Score: {chunk.score:.4f}")

        print()
        print("Preview:")
        print(patent.preview)

        print()
        print("=" * 100)


def main():

    print("\n\033[36m⏳ Loading models...\033[0m")
    start = time.time()
    search = SemanticSearch()
    elapsed = time.time() - start
    print(f"\033[32m✓ Models loaded in {elapsed:.1f}s\033[0m\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_search(search, query)
    else:
        print("=== Interactive Patent Semantic Search ===")
        print("Type your query and press Enter (or type 'exit' or 'q' to quit).\n")

        while True:
            try:
                query = input("\033[1mEnter search query:\033[0m ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "q", "quit"):
                    print("Exiting search.")
                    break
                run_search(search, query)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting search.")
                break


if __name__ == "__main__":
    main()
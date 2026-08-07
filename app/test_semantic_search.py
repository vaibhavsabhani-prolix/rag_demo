import sys
from app.semantic_search import SemanticSearch


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


def run_search(search: SemanticSearch, query: str):
    qdrant_results, reranked_results, results = search.search_detailed(query)

    print()
    print("=" * 100)
    print("Query :", query)
    print("=" * 100)

    # -------------------------------------------------------------
    # Table 1: Qdrant DB Candidate Results (Vector Search)
    # -------------------------------------------------------------
    qdrant_headers = ["#", "Qdrant Score", "Patent ID", "Chunk ID", "Section", "Text Preview"]
    qdrant_rows = []
    qdrant_rank_map = {}

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

    search = SemanticSearch()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_search(search, query)
    else:
        print("\n=== Interactive Patent Semantic Search ===")
        print("Type your query and press Enter (or type 'exit' or 'q' to quit).\n")

        while True:
            try:
                query = input("Enter search query: ").strip()
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
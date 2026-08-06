import sys
from app.semantic_search import SemanticSearch


def run_search(search: SemanticSearch, query: str):
    results = search.search(query)

    print()
    print("=" * 60)
    print("Query :", query)
    print("=" * 60)

    if not results:
        print("\nNo matching patents found.")
        print("=" * 60)
        return

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
        print("=" * 60)


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
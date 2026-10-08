import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.query_understanding.engine import QueryUnderstandingEngine


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Enter your query: ").strip()

    # Validate empty query
    if not query:
        print("Query cannot be empty.")
        return

    # Create query understanding engine
    engine = QueryUnderstandingEngine()

    parsed = engine.parse(
        query,
        use_cache=False,
    )

    # Print the parsed query as formatted JSON
    print("Json has came")


if __name__ == "__main__":
    main()
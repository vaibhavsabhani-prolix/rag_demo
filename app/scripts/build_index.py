"""
Build the chunks collection's search index and show a progress bar until
the collection is GREEN.

The ingest does this itself when it finishes. Run this after an ingest that
crashed (indexing stays paused until then) or to watch a build again — but
not while an ingest is running, since it turns indexing back on:

    PYTHONPATH=. python -m app.scripts.build_index
"""

from app.ingest import wait_for_index
from app.qdrant_db import QdrantDB


def main():
    db = QdrantDB()
    db.resume_indexing()
    wait_for_index(db)


if __name__ == "__main__":
    main()

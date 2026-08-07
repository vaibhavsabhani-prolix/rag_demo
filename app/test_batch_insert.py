from pathlib import Path

from app.parser import PatentParser
from app.chunker import PatentChunker
from app.embedder import Embedder
from app.qdrant_db import QdrantDB


def main():

    db = QdrantDB()

    db.reset_collections()

    parser = PatentParser()
    chunker = PatentChunker()
    embedder = Embedder()

    document = parser.load_patent(
        Path("patents-processed/AP170S1.txt")
    )

    db.upsert_patent_metadata(document.patent_id, document.metadata)

    chunks = chunker.split(document)

    for chunk in chunks:
        embedder.embed(chunk)

    db.insert_batch(chunks)

    print()

    print("Total Points :", db.count_points())


if __name__ == "__main__":
    main()
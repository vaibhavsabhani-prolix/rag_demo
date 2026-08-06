from pathlib import Path

from app.parser import PatentParser
from app.chunker import PatentChunker
from app.embedder import Embedder


def main():

    parser = PatentParser()

    document = parser.load_patent(
        Path("patents-processed/AP170S1.txt")
    )

    chunker = PatentChunker()

    chunks = chunker.split(document)

    embedder = Embedder()

    chunk = embedder.embed(chunks[0])

    print("=" * 60)

    print("Patent ID :", chunk.patent_id)

    print("Chunk ID  :", chunk.chunk_id)

    print()

    print("Vector Length :", len(chunk.vector))

    print()

    print("First 10 Values:")

    print(chunk.vector[:10])


if __name__ == "__main__":
    main()
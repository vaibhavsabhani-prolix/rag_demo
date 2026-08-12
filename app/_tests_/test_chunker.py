from pathlib import Path

from app.chunker import PatentChunker
from app.parser import PatentParser


def main():
    parser = PatentParser()

    document = parser.load_patent(
        Path("patents-processed/AP170S1.txt")
    )

    chunker = PatentChunker()
    chunks = chunker.split(document)

    print("=" * 60)
    print(f"Total Chunks : {len(chunks)}")
    print("=" * 60)

    for chunk in chunks:
        print(f"\nChunk ID : {chunk.chunk_id}")
        print(f"Patent ID : {chunk.patent_id}")
        print("\nText:\n")
        print(chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    main()
import time
from pathlib import Path

from app.chunker import PatentChunker
from app.config import MAX_CHUNK_TOKENS
from app.parser import PatentParser


def main():
    parser = PatentParser()

    document = parser.load_patent(
        Path("patents-processed/AP170S1.txt")
    )
    test_file = Path("patents-processed/AP170S1.txt")
    if not test_file.exists():
        test_file = list(Path("Book-cover-list").glob("*.txt"))[0]

    document = parser.load_patent(test_file)

    chunker = PatentChunker()

    t0 = time.perf_counter()
    chunks = chunker.split(document)
    elapsed = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(f"Patent File  : {test_file.name}")
    print(f"Patent ID    : {document.patent_id}")
    print(f"Total Chunks : {len(chunks)}")
    print(f"Elapsed Time : {elapsed:.2f} ms")
    print("=" * 60)

    for chunk in chunks:
        print(f"\nChunk ID : {chunk.chunk_id}")
        print(f"Patent ID : {chunk.patent_id}")
        print("\nText:\n")
        print(chunk.text)
        print(f"\nChunk ID           : {chunk.chunk_id}")
        print(f"Point ID (UUID)    : {chunk.point_id}")
        print(f"Section            : {chunk.section}")
        print(f"Token Count        : {chunk.token_count} (<= {MAX_CHUNK_TOKENS})")
        print(f"Word Count         : {chunk.word_count}")
        print(f"Doc Chunk Index    : {chunk.document_chunk_index}/{chunk.total_chunks}")
        print(f"Section Chunk Index: {chunk.section_chunk_index}/{chunk.section_total_chunks}")
        print(f"\nText (first 120 chars):\n{chunk.text[:120]}...")
        print("-" * 60)

        assert chunk.token_count <= MAX_CHUNK_TOKENS, f"Chunk {chunk.chunk_id} exceeds {MAX_CHUNK_TOKENS}"
        assert chunk.document_chunk_index == chunk.chunk_id
        assert len(chunk.text) > 0

    print("\nAll chunk assertions passed successfully!")


if __name__ == "__main__":
    main()
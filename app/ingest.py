"""
Patent Ingestion Pipeline
"""

import time
from pathlib import Path

from tqdm import tqdm

from app.chunker import PatentChunker
from app.config import BATCH_SIZE
from app.embedder import Embedder
from app.parser import PatentParser
from app.qdrant_db import QdrantDB


def ingest_directory(directory: str):

    parser = PatentParser()
    chunker = PatentChunker()
    embedder = Embedder()
    db = QdrantDB()

    txt_files = sorted(Path(directory).glob("*.txt"))[101:151]

    print(f"\nFound {len(txt_files)} patent files.\n")

    batch = []

    total_patents = 0
    total_chunks = 0
    failed_patents = 0
    failed_files = []

    start_time = time.time()

    for txt_file in tqdm(txt_files, desc="Indexing Patents"):

        try:

            document = parser.load_patent(txt_file)

            chunks = chunker.split(document)

            total_patents += 1
            total_chunks += len(chunks)

            for chunk in chunks:

                embedder.embed(chunk)

                batch.append(chunk)

            if len(batch) >= BATCH_SIZE:

                db.insert_batch(batch)

                batch.clear()

            # Print progress every 500 patents
            if total_patents % 500 == 0:

                elapsed = time.time() - start_time

                print("\n" + "=" * 60)
                print(f"Processed Patents : {total_patents}/{len(txt_files)}")
                print(f"Chunks Indexed    : {total_chunks}")
                print(f"Failed Patents    : {failed_patents}")
                print(f"Elapsed Time      : {elapsed:.2f} seconds")
                print("=" * 60)

        except Exception as e:

            failed_patents += 1
            failed_files.append(txt_file.name)

            print(f"\nFailed : {txt_file.name}")
            print(f"Reason : {e}")

    # Insert remaining chunks
    if batch:

        db.insert_batch(batch)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)
    print(f"Patents Indexed : {total_patents}")
    print(f"Chunks Indexed  : {total_chunks}")
    print(f"Failed Patents  : {failed_patents}")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    print("=" * 60)

    if failed_files:

        print("\nFailed Files:")

        for file in failed_files:

            print(f"- {file}")


if __name__ == "__main__":

    ingest_directory("us-patent")

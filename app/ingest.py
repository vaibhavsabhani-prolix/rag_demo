"""
Patent Ingestion Pipeline

Flow (unchanged): parse -> store patent metadata -> chunk -> embed ->
batch-insert chunk vectors into Qdrant.

The work is the same; it is scheduled better. Three things dominated
the original runtime:

    1. Chunks were embedded one at a time. Embedding is by far the most
       expensive stage, and a single chunk cannot fill the model's
       matrix multiplies. Chunks are now embedded per patent in one
       batched forward pass (Embedder.embed_batch).

    2. Every patent cost its own HTTP round trip to write metadata.
       Metadata points are now buffered and flushed in groups
       (QdrantDB.upsert_patent_metadata_batch).

    3. Reading and chunking a patent happened between embedding calls,
       so the CPU alternated between the two instead of doing them at
       once. A small prefetch thread now parses and chunks upcoming
       patents while the model embeds the current one.
"""

import time
from pathlib import Path
from queue import Queue
from threading import Thread

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from app.chunker import PatentChunker
from app.config import (
    BATCH_SIZE,
    INGEST_PREFETCH,
    METADATA_BATCH_SIZE,
    PATENT_DIRECTORY,
)
from app.embedder import Embedder
from app.parser import PatentParser
from app.qdrant_db import QdrantDB

# Sentinel pushed onto the prefetch queue to mark end-of-input.
_DONE = object()


class RateColumn(ProgressColumn):
    """
    Throughput column for the ingest bar.

    Rich ships speed columns for byte transfers, not for items, so this
    fills the gap. Ingestion normally runs at well under one patent per
    second, where "0.04/s" is harder to read than the time per patent,
    so the unit is inverted below 1/s.
    """

    def render(self, task) -> Text:

        speed = task.finished_speed or task.speed

        if not speed:
            return Text("--", style="progress.data.speed")

        if speed >= 1:
            return Text(f"{speed:.2f}/s", style="progress.data.speed")

        return Text(f"{1 / speed:.1f}s each", style="progress.data.speed")


def _build_progress(quiet: bool = False) -> Progress:
    """
    Build the ingest progress display.

    Rich detects a redirected stdout on its own and skips the live
    redraw, so a multi-hour ingest piped to a log file stays readable
    instead of collecting thousands of control characters.
    """

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description:<17}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("-"),
        RateColumn(),
        TextColumn("- elapsed"),
        TimeElapsedColumn(),
        TextColumn("- eta"),
        TimeRemainingColumn(),
        disable=quiet,
    )


def _prefetch_documents(txt_files, queue: Queue):
    """
    Parse and chunk patents ahead of the embedder, on a worker thread.

    Produces (txt_file, document, chunks, error) tuples in the SAME
    order as *txt_files*, so ingestion order and per-file error
    reporting are identical to processing them inline.

    A bounded queue keeps this from running away: the prefetcher blocks
    once it is INGEST_PREFETCH patents ahead, so memory stays flat no
    matter how large the directory is.
    """

    parser = PatentParser()
    chunker = PatentChunker()

    for txt_file in txt_files:
        try:
            document = parser.load_patent(txt_file)
            chunks = chunker.split(document)

            queue.put((txt_file, document, chunks, None))

        except Exception as exc:
            queue.put((txt_file, None, None, exc))

    queue.put(_DONE)


def ingest_directory(directory: str):

    embedder = Embedder()
    db = QdrantDB()
    db.create_collections()

    txt_files = sorted(Path(directory).glob("*.txt"))

    print(f"\nFound {len(txt_files)} patent files.\n")

    # Chunks waiting to be written, and the patents they belong to.
    # Flushes only ever happen on a patent boundary, so every patent
    # listed here has all of its chunks in the batch - which is what
    # makes the per-patent "inserted" count below exact.
    batch = []
    pending_patents = []

    metadata_batch = []

    total_patents = 0
    total_chunks = 0
    total_inserted = 0
    failed_patents = 0
    failed_files = []

    start_time = time.time()

    # Bounded hand-off queue: parse/chunk runs ahead of embedding by at
    # most INGEST_PREFETCH patents.
    queue: Queue = Queue(maxsize=INGEST_PREFETCH)

    producer = Thread(
        target=_prefetch_documents,
        args=(txt_files, queue),
        daemon=True,
    )
    producer.start()

    progress = _build_progress()

    def report(message: str):
        """
        Print above the live bar.

        markup/highlight are off so a patent ID or an exception message
        containing square brackets is shown verbatim instead of being
        parsed as Rich markup.
        """

        progress.console.print(message, markup=False, highlight=False)

    def flush(wait: bool = False):
        """Write buffered metadata + chunks, then report per patent."""

        nonlocal total_inserted

        if metadata_batch:
            db.upsert_patent_metadata_batch(metadata_batch, wait=wait)
            metadata_batch.clear()

        if batch:
            total_inserted += db.insert_batch(batch, wait=wait)
            batch.clear()

        # Reported after the write, so a patent is only ever announced
        # as inserted once its chunks have actually gone to Qdrant. A
        # patent that produced no chunks still reports, as 0 / 0.
        for name, embedded_count in pending_patents:
            report(
                f"{name:<20} | embedded {embedded_count:>4} chunks"
                f" | inserted {embedded_count:>4} chunks"
            )

        pending_patents.clear()

    with progress:
        patent_task = progress.add_task("Indexing Patents", total=len(txt_files))

        # Total chunk count is not knowable up front - it is discovered
        # a patent at a time - so this task starts indeterminate and its
        # total grows as the prefetcher reports what each patent split
        # into.
        chunk_task = progress.add_task("Embedding Chunks", total=None)

        discovered_chunks = 0

        while True:
            item = queue.get()

            if item is _DONE:
                break

            txt_file, document, chunks, error = item

            # Advanced in the finally block, once the patent is actually
            # finished. Advancing on dequeue would report a patent as
            # done before a single one of its chunks had been embedded.
            try:
                if error is not None:
                    failed_patents += 1
                    failed_files.append(txt_file.name)

                    report(f"\nFailed : {txt_file.name}")
                    report(f"Reason : {error}")
                    continue

                # Parsing and chunking already succeeded on the prefetch
                # thread; this guards the embed/write half, so one bad
                # patent still cannot abort the run.
                try:
                    discovered_chunks += len(chunks)
                    progress.update(chunk_task, total=discovered_chunks)

                    report(
                        f"{document.patent_id:<20} | split into"
                        f" {len(chunks):>5} chunks - embedding"
                    )

                    # Metadata is buffered rather than written per
                    # patent, but still written exactly once per patent,
                    # before its chunks.
                    metadata_batch.append(
                        (document.patent_id, document.metadata)
                    )

                    # Batched forward passes instead of one call per
                    # chunk, reporting movement as each batch lands so a
                    # patent with thousands of chunks still shows life.
                    embedder.embed_batch(
                        chunks,
                        on_progress=lambda done: progress.advance(
                            chunk_task, done
                        ),
                    )

                    batch.extend(chunks)
                    pending_patents.append((document.patent_id, len(chunks)))

                    total_patents += 1
                    total_chunks += len(chunks)

                    if (
                        len(batch) >= BATCH_SIZE
                        or len(metadata_batch) >= METADATA_BATCH_SIZE
                    ):
                        flush()

                except Exception as exc:
                    failed_patents += 1
                    failed_files.append(txt_file.name)

                    report(f"\nFailed : {txt_file.name}")
                    report(f"Reason : {exc}")
                    continue

            finally:
                progress.advance(patent_task)

            # Print progress every 500 patents
            if total_patents % 500 == 0:
                elapsed = time.time() - start_time

                report("\n" + "=" * 60)
                report(f"Processed Patents : {total_patents}/{len(txt_files)}")
                report(f"Chunks Indexed    : {total_chunks}")
                report(f"Failed Patents    : {failed_patents}")
                report(f"Elapsed Time      : {elapsed:.2f} seconds")
                report(f"Rate              : {total_patents / elapsed:.2f} patents/s")
                report("=" * 60)

    producer.join()

    # Insert remaining chunks. The final flush waits for Qdrant to
    # acknowledge, so the counts printed below reflect committed data.
    flush(wait=True)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)
    print(f"Patents Indexed : {total_patents}")
    print(f"Chunks Indexed  : {total_chunks}")
    print(f"Chunks Inserted : {total_inserted}")
    print(f"Failed Patents  : {failed_patents}")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    print("=" * 60)

    if failed_files:
        print("\nFailed Files:")

        for file in failed_files:
            print(f"- {file}")


if __name__ == "__main__":
    ingest_directory(PATENT_DIRECTORY)

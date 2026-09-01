import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

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
    EMBED_BATCH_SIZE,
    INGEST_PREFETCH,
    INGEST_PROGRESS_FILE,
    INSERT_REPORT_EVERY,
    METADATA_BATCH_SIZE,
    PATENT_DIRECTORY,
)
from app.embedder import Embedder
from app.parser import PatentParser
from app.qdrant_db import QdrantDB

# Sentinels pushed onto the prefetch/write queues to mark end-of-input.
_DONE = object()
_WRITE_DONE = object()

# How many completed batches may be waiting on the writer thread at
# once: one uploading plus one queued behind it. This decouples
# embedding from the write's HTTP round trip without letting embedded
# chunks pile up in memory without limit if Qdrant falls behind.
_WRITE_QUEUE_DEPTH = 2


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


def _prefetch_documents(txt_files, queue: Queue, on_progress=None):
    """
    Parse and chunk patents ahead of the embedder, using a ThreadPoolExecutor
    across 16 parallel threads to utilize all CPU cores.

    Produces (txt_file, document, chunks, error) tuples in the SAME
    order as *txt_files*, so ingestion order and per-file error
    reporting are identical to processing them inline.

    A bounded queue keeps this from running away: the prefetcher blocks
    once it is INGEST_PREFETCH patents ahead, so memory stays flat no
    matter how large the directory is.
    """

    parser = PatentParser()
    chunker = PatentChunker()

    def _process_one(txt_file):
        try:
            document = parser.load_patent(txt_file)
            chunks = chunker.split(document)
            return (txt_file, document, chunks, None)
        except Exception as exc:
            return (txt_file, None, None, exc)

    workers = max(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_process_one, txt_files):
            queue.put(result)
            if on_progress is not None:
                on_progress(1)

    queue.put(_DONE)


def _load_completed_patents(path: Path) -> set[str]:
    """
    Read the set of patent filenames already fully committed to Qdrant
    from a previous run, so ingest_directory() can skip them.

    Missing file means nothing has ever been checkpointed - a fresh
    directory, not an error.
    """

    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def ingest_directory(directory: str):

    embedder = Embedder()
    db = QdrantDB()
    db.create_collections()

    all_txt_files = sorted(Path(directory).glob("*.txt"))

    progress_path = Path(INGEST_PROGRESS_FILE)
    completed_patents = _load_completed_patents(progress_path)

    txt_files = [f for f in all_txt_files if f.name not in completed_patents]
    skipped = len(all_txt_files) - len(txt_files)

    if skipped:
        print(f"\nResuming: {skipped} patent(s) already completed, skipping.")

    print(f"\nFound {len(txt_files)} patent file(s) left to process.\n")
    print("Starting ingestion: scanning/chunking and embedding concurrently...\n")

    # Opened once and appended to as each batch is confirmed fully
    # written (see flush() below), so a stopped or crashed run can be
    # resumed by re-running the same command: the patents already
    # recorded here are skipped on the next start instead of redone.
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = open(progress_path, "a", encoding="utf-8")

    # Chunks waiting to be written, and the patents they belong to.
    # Once a batch is handed to the writer thread (see below) these
    # are replaced with fresh lists rather than cleared in place,
    # since the writer may still be reading the ones just handed off.
    batch = []
    pending_patents = []

    metadata_batch = []

    # Cross-patent embed pool: chunks from multiple patents are
    # accumulated here until the pool reaches EMBED_BATCH_SIZE, then
    # embedded in one call. This fills every GPU batch completely even
    # when individual patents are small (5-50 chunks each), instead of
    # sending half-empty requests per patent.
    embed_pool = []
    pool_patents = []  # (patent_id, chunk_count, filename) per pooled patent

    total_patents = 0
    total_chunks = 0
    total_inserted = 0
    failed_patents = 0
    failed_files = []

    # Patents that embedded fine but whose batch write was rejected.
    # Kept apart from failed_files: those failed before producing
    # anything, these produced chunks that simply never landed.
    write_failures = []

    start_time = time.time()

    # Bounded hand-off queue: parse/chunk runs ahead of embedding by at
    # most INGEST_PREFETCH patents.
    queue: Queue = Queue(maxsize=INGEST_PREFETCH)

    scan_task = None

    def on_scan_progress(done: int = 1):
        if scan_task is not None:
            advance(scan_task, done)

    producer = Thread(
        target=_prefetch_documents,
        args=(txt_files, queue, on_scan_progress),
        daemon=True,
    )
    producer.start()

    # Bounded hand-off to the writer thread, mirroring the prefetch
    # queue above but on the output side: the main loop keeps
    # embedding the next patent while a previous batch uploads to
    # Qdrant on this separate thread instead of blocking on it.
    write_queue: Queue = Queue(maxsize=_WRITE_QUEUE_DEPTH)

    progress = _build_progress()

    # Guards every touch of `progress` (and its console): embedding
    # progress advances from the main thread, insert progress advances
    # from the writer thread, and Rich's Progress makes no promise
    # about concurrent callers.
    progress_lock = Lock()

    # Assigned once the progress block below is open. flush() reads it
    # as a closure, and is never called before it is set.
    insert_task = None

    def report(message: str):
        """
        Print above the live bar.

        markup/highlight are off so a patent ID or an exception message
        containing square brackets is shown verbatim instead of being
        parsed as Rich markup.
        """

        with progress_lock:
            progress.console.print(message, markup=False, highlight=False)

    def advance(task, amount: int = 1):
        with progress_lock:
            progress.advance(task, amount)

    def flush(metadata_batch, batch, pending_patents, wait: bool = False):
        """
        Write one already-assembled batch of metadata + chunks to
        Qdrant, then report per patent.

        Runs on the writer thread only - the main loop hands off a
        batch and immediately starts a fresh one rather than calling
        this directly, so embedding the next patent and writing the
        previous one happen at the same time instead of one blocking
        the other.

        The error is reported and swallowed for the same reason as
        before: the patents in this batch are lost either way, and the
        run has no reason to stop writing the ones that follow.
        """

        nonlocal total_inserted

        write_error = None

        pending_chunks = len(batch)
        pending_metadata = len(metadata_batch)

        if pending_chunks or pending_metadata:
            report(
                f"{'writing to qdrant':<20} | {pending_chunks:>5} chunks"
                f" | {pending_metadata:>4} patent metadata"
            )

        # Chunks written so far in THIS flush, and the run-wide count
        # at which the next live line is due. The threshold is carried
        # across flushes rather than reset per flush: a flush is
        # usually only one or two Qdrant requests, so a per-flush
        # counter never reached the reporting interval and the lines
        # only ever showed up on the rare patent big enough to need
        # thousands of writes on its own.
        written = 0
        next_report = total_inserted + INSERT_REPORT_EVERY

        def on_written(done: int):
            nonlocal written, next_report, total_inserted

            written += done

            # Counted here rather than from insert_batch's return
            # value, so the running total moves while the write is
            # still going - and so a batch that fails halfway still
            # counts the requests that did land, instead of discarding
            # them along with the ones that did not.
            total_inserted += done

            advance(insert_task, done)

            if total_inserted >= next_report:
                report(
                    f"{'inserting':<20} | {written:>6} /"
                    f" {pending_chunks} chunks written"
                    f" | running total {total_inserted}"
                )

                next_report = total_inserted + INSERT_REPORT_EVERY

        try:
            if metadata_batch:
                db.upsert_patent_metadata_batch(metadata_batch, wait=wait)

            if batch:
                # Return value ignored: on_written already counted
                # every point as its request landed.
                db.insert_batch(
                    batch,
                    wait=wait,
                    on_progress=on_written,
                )

        except Exception as exc:
            write_error = exc

        # Reported after the write, so a patent is only ever announced
        # as inserted once its chunks have actually gone to Qdrant. A
        # patent that produced no chunks still reports, as 0 / 0.
        if write_error is None:
            if pending_chunks:
                report(
                    f"{'inserted':<20} | {pending_chunks:>6} chunks committed"
                    f" to qdrant | running total {total_inserted}"
                )

            # Checkpointed only here, once metadata + chunks are both
            # confirmed committed - never on a failed or partial write,
            # so a resumed run always retries anything not fully done
            # rather than treating a partial batch as complete.
            for name, embedded_count, filename in pending_patents:
                report(
                    f"{name:<20} | embedded {embedded_count:>4} chunks"
                    f" | inserted {embedded_count:>4} chunks"
                )

                progress_file.write(filename + "\n")

            progress_file.flush()
        else:
            report(f"\nFailed to write batch: {write_error}")

            for name, embedded_count, filename in pending_patents:
                report(
                    f"{name:<20} | embedded {embedded_count:>4} chunks | NOT inserted"
                )

                write_failures.append(name)

    def writer_loop():
        """
        Drain write_queue and flush each batch, one at a time.

        A single consumer thread, so batches land at Qdrant in the
        same order the main loop produced them, without needing a
        lock around total_inserted/write_failures - only this thread
        ever touches them.
        """

        while True:
            job = write_queue.get()

            if job is _WRITE_DONE:
                break

            flush(*job)

    writer = Thread(target=writer_loop, daemon=True)
    writer.start()

    with progress:
        scan_task = progress.add_task("Scanning Patents", total=len(txt_files))
        patent_task = progress.add_task("Indexing Patents", total=len(txt_files))

        chunk_task = progress.add_task("Embedding Chunks", total=None)
        insert_task = progress.add_task("Inserting Chunks", total=None)

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
                    report(
                        f"{document.patent_id:<20} | split into"
                        f" {len(chunks):>5} chunks - pooling"
                    )

                    # Metadata is buffered rather than written per
                    # patent, but still written exactly once per patent,
                    # before its chunks.
                    metadata_batch.append((document.patent_id, document.metadata))

                    # Pool chunks for cross-patent batching: small
                    # patents (5-50 chunks) are accumulated so each GPU
                    # request carries a full EMBED_BATCH_SIZE batch
                    # instead of a mostly-empty one.
                    embed_pool.extend(chunks)
                    pool_patents.append(
                        (document.patent_id, len(chunks), txt_file.name)
                    )

                    total_patents += 1
                    total_chunks += len(chunks)

                    with progress_lock:
                        progress.update(chunk_task, total=total_chunks)
                        progress.update(insert_task, total=total_chunks)

                    # Flush the pool once it has enough chunks to fill
                    # at least one full GPU batch.
                    if len(embed_pool) >= EMBED_BATCH_SIZE:
                        embedder.embed_batch(
                            embed_pool,
                            on_progress=lambda done: advance(chunk_task, done),
                        )

                        batch.extend(embed_pool)
                        pending_patents.extend(pool_patents)
                        embed_pool = []
                        pool_patents = []

                    if (
                        len(batch) >= BATCH_SIZE
                        or len(metadata_batch) >= METADATA_BATCH_SIZE
                    ):
                        # Hand the completed batch to the writer thread
                        # and start fresh ones immediately, so the next
                        # patent's chunks embed while this batch uploads
                        # instead of waiting for it to finish.
                        write_queue.put((metadata_batch, batch, pending_patents, False))
                        batch = []
                        pending_patents = []
                        metadata_batch = []

                except Exception as exc:
                    # Embedding failure affects every patent in the pool.
                    for pool_pid, pool_nc, pool_fn in pool_patents:
                        failed_patents += 1
                        failed_files.append(pool_fn)
                        report(f"\nFailed : {pool_fn}")

                    report(f"Reason : {exc}")
                    embed_pool = []
                    pool_patents = []
                    continue

            finally:
                advance(patent_task)

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

    # Embed any remaining chunks left in the pool after the last patent
    # was dequeued — the pool only flushes once it reaches
    # EMBED_BATCH_SIZE, so the tail end of a run will always have a
    # partial pool unless the total chunk count happens to be a multiple.
    if embed_pool:
        try:
            embedder.embed_batch(embed_pool)
            batch.extend(embed_pool)
            pending_patents.extend(pool_patents)
        except Exception as exc:
            for pool_pid, pool_nc, pool_fn in pool_patents:
                failed_patents += 1
                failed_files.append(pool_fn)
            print(f"\nFailed to embed final pool: {exc}")

    # Final batch, written with wait=True so the counts below reflect
    # data Qdrant has actually committed, then shut the writer thread
    # down behind it.
    write_queue.put((metadata_batch, batch, pending_patents, True))
    write_queue.put(_WRITE_DONE)
    writer.join()

    progress_file.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)
    print(f"Patents Indexed : {total_patents}")
    print(f"Chunks Indexed  : {total_chunks}")
    print(f"Chunks Inserted : {total_inserted}")
    print(f"Failed Patents  : {failed_patents}")
    print(f"Write Failures  : {len(write_failures)}")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    print("=" * 60)

    if failed_files:
        print("\nFailed Files:")

        for file in failed_files:
            print(f"- {file}")

    # Distinct from the list above: these parsed, chunked and embedded
    # successfully, and were lost at the write. Re-running them costs
    # only the write, not the embedding, if the cause is fixed first.
    if write_failures:
        print("\nEmbedded But Not Inserted:")

        for name in write_failures:
            print(f"- {name}")


if __name__ == "__main__":
    ingest_directory(PATENT_DIRECTORY)

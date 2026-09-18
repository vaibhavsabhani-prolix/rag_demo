"""
Concurrent Patent Ingestion

Five stages, each with its own worker pool, connected by bounded
queues so a slow stage backs up the one feeding it instead of one
stage's I/O stalling another:

    parallel parsing/chunking workers   parse + chunk patent files
        -> parsed_queue (patent-level, one entry per patent)
    chunk dispatcher (main thread)      buffers raw chunks. The first
                                        CHUNK_QUEUE_CAPACITY chunks
                                        accumulate before any embedding
                                        window is cut, so the first
                                        round of requests goes out
                                        against a full backlog instead
                                        of dribbling out while parsing
                                        is still warming up. After
                                        that, one window is cut every
                                        time >= EMBED_BATCH_SIZE chunks
                                        are buffered.
        -> embed_task_queue
    EMBED_CONCURRENT_REQUESTS           long-lived pool: each worker
    embedding workers                   embeds one window, then loops
                                        straight back for the next
                                        ready one - no batch boundary
                                        to wait out.
        -> vector_queue
    vector dispatcher (thread)          buffers embedded windows; once
                                        BATCH_SIZE chunks are on hand,
                                        cuts an insertion batch.
        -> insertion_task_queue
    INSERT_WORKERS                      each upserts one batch (chunk
    insert workers                      vectors + patent metadata) to
                                        Qdrant and checkpoints the
                                        patents it just committed.

Concurrency is controlled by four config values: CHUNK_QUEUE_CAPACITY,
EMBED_BATCH_SIZE, EMBED_CONCURRENT_REQUESTS and INSERT_WORKERS - all in
app/config.py.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

from app.chunker import PatentChunker
from app.config import (
    BATCH_SIZE,
    CHUNK_QUEUE_CAPACITY,
    EMBED_BATCH_SIZE,
    EMBED_CONCURRENT_REQUESTS,
    INGEST_PREFETCH,
    INGEST_PROGRESS_FILE,
    INSERT_REPORT_EVERY,
    INSERT_WORKERS,
    PATENT_DIRECTORY,
)
from app.embedder import Embedder
from app.parser import PatentParser
from app.progress import IngestProgress
from app.qdrant_db import QdrantDB

_DONE = object()
_EMBED_DONE = object()
_INSERT_DONE = object()

# Windows waiting on an embed worker, sized off the same
# CHUNK_QUEUE_CAPACITY the chunk dispatcher primes against, so the
# first priming flush (CHUNK_QUEUE_CAPACITY / EMBED_BATCH_SIZE windows)
# always fits without blocking on a worker completing first.
_EMBED_QUEUE_DEPTH = max(EMBED_CONCURRENT_REQUESTS, CHUNK_QUEUE_CAPACITY // EMBED_BATCH_SIZE)

# Insertion batches waiting on an insert worker, mirroring the embed
# queue's headroom on the write side.
_INSERT_QUEUE_DEPTH = INSERT_WORKERS * 2


def _scan_patent_files(directory: str) -> list[Path]:
    """List every *.txt patent file in *directory* using a directory scanner."""

    with os.scandir(directory) as entries:
        return sorted(
            Path(entry.path)
            for entry in entries
            if entry.is_file() and entry.name.endswith(".txt")
        )


def _load_completed_patents(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def _prefetch_documents(txt_files, parsed_queue: Queue, on_progress=None):
    """
    Parse and chunk patents on a pool of worker threads, pushing
    (path, document, chunks, error) tuples onto *parsed_queue* as each
    one finishes.
    """

    parser = PatentParser()
    chunker = PatentChunker()

    def process_one(path):
        try:
            document = parser.load_patent(path)
            chunks = chunker.split(document)
            return path, document, chunks, None
        except Exception as exc:
            return path, None, None, exc

    workers = max(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_one, txt_files):
            parsed_queue.put(result)
            if on_progress is not None:
                on_progress(1)

    parsed_queue.put(_DONE)


def _embed_worker(
    embedder: Embedder,
    embed_task_queue: Queue,
    vector_queue: Queue,
    progress: IngestProgress,
    failures_lock: Lock,
    failed_files: list,
):
    """
    Pull the next ready embedding window and embed it, forever, until
    told to stop via _EMBED_DONE. A fixed pool of these run against the
    same queue for the whole run, so the moment one finishes it is
    straight back at queue.get() for whatever window is next.
    """

    while True:
        job = embed_task_queue.get()

        if job is _EMBED_DONE:
            return

        window_chunks, window_metadata, window_patents, window_number = job

        try:
            embedder.embed_batch(
                window_chunks,
                on_progress=progress.advance_embedding,
                label=f"window {window_number}",
            )
        except Exception as exc:
            progress.report(f"Embedding failure: {exc}")

            with failures_lock:
                for _patent_id, _count, filename in window_patents:
                    failed_files.append(filename)

            continue

        vector_queue.put((window_metadata, window_chunks, window_patents))


class _VectorDispatcher:
    """
    Buffers embedded windows as embed workers finish them and cuts an
    insertion batch every time BATCH_SIZE chunks are on hand.

    Runs on a single thread so the buffer never needs a lock: only
    this dispatcher ever touches it, the same way the chunk dispatcher
    below owns the pre-embedding buffer.
    """

    def __init__(self, vector_queue: Queue, insertion_task_queue: Queue):
        self.vector_queue = vector_queue
        self.insertion_task_queue = insertion_task_queue

        self._metadata_buffer = []
        self._chunk_buffer = []
        self._pending_buffer = []

    def _submit(self, wait: bool):
        if not self._metadata_buffer and not self._chunk_buffer:
            return

        self.insertion_task_queue.put(
            (self._metadata_buffer, self._chunk_buffer, self._pending_buffer, wait)
        )
        self._metadata_buffer = []
        self._chunk_buffer = []
        self._pending_buffer = []

    def run(self):
        while True:
            job = self.vector_queue.get()

            if job is _DONE:
                # Final partial batch, written with wait=True so the
                # caller can trust the counts once this returns.
                self._submit(wait=True)
                return

            metadata, chunks, pending_patents = job
            self._metadata_buffer.extend(metadata)
            self._chunk_buffer.extend(chunks)
            self._pending_buffer.extend(pending_patents)

            if len(self._chunk_buffer) >= BATCH_SIZE:
                self._submit(wait=False)


class _InsertStats:
    """Counters shared across the parallel insert workers, behind a lock."""

    def __init__(self):
        self._lock = Lock()
        self.total_inserted = 0
        self.write_failures = []
        self._next_report = INSERT_REPORT_EVERY

    def record_inserted(self, count: int, progress: IngestProgress):
        with self._lock:
            self.total_inserted += count
            progress.advance_inserting(count)

            due = self.total_inserted >= self._next_report
            if due:
                self._next_report = self.total_inserted + INSERT_REPORT_EVERY
                total = self.total_inserted

        if due:
            progress.report(f"inserting | total={total}")

    def record_failure(self, patent_ids):
        with self._lock:
            self.write_failures.extend(patent_ids)


def _insert_worker(
    db: QdrantDB,
    insertion_task_queue: Queue,
    stats: _InsertStats,
    progress: IngestProgress,
    progress_file,
    progress_file_lock: Lock,
):
    """
    Pull one assembled batch at a time and upsert it to Qdrant. A pool
    of these run against the same queue, so a slow upsert only blocks
    the worker handling it, not the batches queued behind it.
    """

    while True:
        job = insertion_task_queue.get()

        if job is _INSERT_DONE:
            return

        metadata, chunks, pending_patents, wait = job

        if not metadata and not chunks:
            continue

        try:
            if metadata:
                db.upsert_patent_metadata_batch(metadata, wait=wait)

            if chunks:
                db.insert_batch(
                    chunks,
                    batch_size=BATCH_SIZE,
                    on_progress=lambda n: stats.record_inserted(n, progress),
                    wait=wait,
                )

            with progress_file_lock:
                for patent_id, count, filename in pending_patents:
                    progress.report(
                        f"{patent_id:<20} | embedded={count:>4} | inserted={count:>4}"
                    )
                    progress_file.write(filename + "\n")
                    progress.advance_completed()
                progress_file.flush()

        except Exception as exc:
            progress.report(f"Write failure: {exc}")
            stats.record_failure(patent_id for patent_id, _, _ in pending_patents)


def ingest_directory(directory: str):
    embedder = Embedder()
    db = QdrantDB()
    db.create_collections()

    all_files = _scan_patent_files(directory)
    progress_path = Path(INGEST_PROGRESS_FILE)
    completed = _load_completed_patents(progress_path)
    txt_files = [path for path in all_files if path.name not in completed]

    print(f"Found {len(txt_files)} file(s) to process.")
    print(f"Skipped {len(all_files) - len(txt_files)} completed file(s).")

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = progress_path.open("a", encoding="utf-8")
    progress_file_lock = Lock()

    progress = IngestProgress(total_files=len(txt_files))

    parsed_queue: Queue = Queue(maxsize=INGEST_PREFETCH)
    embed_task_queue: Queue = Queue(maxsize=_EMBED_QUEUE_DEPTH)
    vector_queue: Queue = Queue()
    insertion_task_queue: Queue = Queue(maxsize=_INSERT_QUEUE_DEPTH)

    failures_lock = Lock()
    failed_files: list[str] = []
    stats = _InsertStats()

    # ---- Stage 5: parallel Qdrant insert workers ----
    insert_workers = [
        Thread(
            target=_insert_worker,
            args=(db, insertion_task_queue, stats, progress, progress_file, progress_file_lock),
            daemon=True,
        )
        for _ in range(INSERT_WORKERS)
    ]
    for worker in insert_workers:
        worker.start()

    # ---- Stage 4: vector dispatcher (buffers embedded chunks into insertion batches) ----
    vector_dispatcher = _VectorDispatcher(vector_queue, insertion_task_queue)
    vector_thread = Thread(target=vector_dispatcher.run, daemon=True)
    vector_thread.start()

    # ---- Stage 3: parallel embedding workers ----
    embed_workers = [
        Thread(
            target=_embed_worker,
            args=(embedder, embed_task_queue, vector_queue, progress, failures_lock, failed_files),
            daemon=True,
        )
        for _ in range(EMBED_CONCURRENT_REQUESTS)
    ]
    for worker in embed_workers:
        worker.start()

    # ---- Stage 1: parallel parsing/chunking workers ----
    producer_thread = Thread(
        target=_prefetch_documents,
        args=(txt_files, parsed_queue, progress.advance_scanning),
        daemon=True,
    )
    producer_thread.start()

    # ---- Stage 2: chunk dispatcher (this thread) ----
    # Raw chunks buffered ahead of the first embedding window. Not
    # flushed at all until it reaches CHUNK_QUEUE_CAPACITY for the
    # first time, so the first round of embedding requests goes out
    # against a full backlog instead of a few half-empty ones while
    # parsing is still ramping up.
    #
    # Windows are cut at exact EMBED_BATCH_SIZE offsets, not patent
    # boundaries, so CHUNK_QUEUE_CAPACITY buffered chunks always yields
    # exactly CHUNK_QUEUE_CAPACITY / EMBED_BATCH_SIZE clean windows (one
    # HTTP request each) instead of oversized ones that
    # embedder.embed_batch then has to re-split. A patent's chunks can
    # end up split across two windows; buffer_owners tracks which
    # patent each buffered chunk belongs to, and each patent_table
    # entry's "remaining" counts down so a patent is only checkpointed
    # once its LAST chunk has actually gone out, whichever window that
    # lands in.
    buffer_chunks = []
    buffer_owners = []  # same length as buffer_chunks: index into patent_table
    patent_table = []  # index -> {"patent_id", "metadata", "filename", "count", "remaining"}
    primed = False

    # Sequence number stamped on each window as it's cut, purely for
    # the request log - so consecutive requests read "window 1",
    # "window 2", ... instead of embed_batch's default "1/1" on every
    # single one now that windows are always <= EMBED_BATCH_SIZE.
    window_counter = 0

    def submit_window():
        """Cut exactly one EMBED_BATCH_SIZE window (or the final partial one) off the front of the buffer."""

        nonlocal buffer_chunks, buffer_owners, window_counter

        if not buffer_chunks:
            return

        n = min(EMBED_BATCH_SIZE, len(buffer_chunks))
        window_chunks = buffer_chunks[:n]
        window_owners = buffer_owners[:n]
        buffer_chunks = buffer_chunks[n:]
        buffer_owners = buffer_owners[n:]

        window_metadata = []
        window_patents = []
        completed = set()

        for owner in window_owners:
            entry = patent_table[owner]
            entry["remaining"] -= 1
            if entry["remaining"] == 0 and owner not in completed:
                completed.add(owner)
                window_metadata.append((entry["patent_id"], entry["metadata"]))
                window_patents.append(
                    (entry["patent_id"], entry["count"], entry["filename"])
                )

        window_counter += 1

        # Blocks only once every embed worker already has a window
        # queued - the backpressure that keeps memory flat.
        embed_task_queue.put(
            (window_chunks, window_metadata, window_patents, window_counter)
        )

    total_patents = 0
    total_chunks = 0
    start_time = time.time()

    with progress:
        while True:
            item = parsed_queue.get()
            if item is _DONE:
                break

            path, document, chunks, error = item

            if error is not None:
                with failures_lock:
                    failed_files.append(path.name)
                progress.report(f"Failed: {path.name} | {error}")
                continue

            count = len(chunks)
            patent_id = document.patent_id

            progress.advance_chunking(count)
            progress.report(f"{patent_id:<20} | chunks={count:>4} | buffering")

            owner = len(patent_table)
            patent_table.append(
                {
                    "patent_id": patent_id,
                    "metadata": document.metadata,
                    "filename": path.name,
                    "count": count,
                    "remaining": count,
                }
            )
            buffer_chunks.extend(chunks)
            buffer_owners.extend([owner] * count)

            total_patents += 1
            total_chunks += count

            if not primed:
                if len(buffer_chunks) >= CHUNK_QUEUE_CAPACITY:
                    primed = True
                    # Cut every full window out of the now-primed
                    # buffer in one go, so all embed workers have work
                    # waiting immediately instead of racing the
                    # dispatcher one window at a time.
                    while len(buffer_chunks) >= EMBED_BATCH_SIZE:
                        submit_window()
            else:
                while len(buffer_chunks) >= EMBED_BATCH_SIZE:
                    submit_window()

            if total_patents % 500 == 0:
                elapsed = max(time.time() - start_time, 0.001)
                progress.report(
                    f"Processed={total_patents} | chunks={total_chunks} | "
                    f"rate={total_patents / elapsed:.2f} patents/s"
                )

    producer_thread.join()

    # Whatever's left in the buffer, primed or not, still needs
    # embedding - flush it out as however many windows it makes.
    while buffer_chunks:
        submit_window()

    for _ in embed_workers:
        embed_task_queue.put(_EMBED_DONE)
    for worker in embed_workers:
        worker.join()

    # No more embedded windows are coming - let the vector dispatcher
    # cut its final (possibly partial) batch and stop.
    vector_queue.put(_DONE)
    vector_thread.join()

    for _ in insert_workers:
        insertion_task_queue.put(_INSERT_DONE)
    for worker in insert_workers:
        worker.join()

    embedder.close()
    progress_file.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED")
    print("=" * 60)
    print(f"Patents Indexed : {total_patents}")
    print(f"Chunks Indexed  : {total_chunks}")
    print(f"Chunks Inserted : {stats.total_inserted}")
    print(f"Failed Patents  : {len(failed_files)}")
    print(f"Write Failures  : {len(stats.write_failures)}")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    print("=" * 60)

    if failed_files:
        print("\nFailed Files:")
        for filename in failed_files:
            print(f"- {filename}")

    if stats.write_failures:
        print("\nEmbedded But Not Inserted:")
        for patent_id in stats.write_failures:
            print(f"- {patent_id}")


if __name__ == "__main__":
    ingest_directory(PATENT_DIRECTORY)

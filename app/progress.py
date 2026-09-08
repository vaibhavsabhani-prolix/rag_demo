"""
Ingest Progress Display

All Rich progress-bar logic for the ingestion pipeline lives here,
keeping ingest.py focused on the pipeline itself.

Five bars track each stage of the pipeline:

    1. Scanning Patents   — files parsed + chunked by the prefetch thread
    2. Chunking           — total chunks discovered (grows as scanning reveals them)
    3. Embedding Chunks   — chunks embedded by the GPU
    4. Inserting Chunks   — chunks written to Qdrant
    5. Completed Patents  — patents fully committed + checkpointed

SmartETAColumn provides accurate time-remaining estimates using a
rolling window of recent completion timestamps, avoiding the
inaccuracies that Rich's built-in TimeRemainingColumn shows when
totals change mid-task.
"""

from __future__ import annotations

import collections
import time
from threading import Lock

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


# ==============================================================
# Custom columns
# ==============================================================


class RateColumn(ProgressColumn):
    """
    Throughput column for ingest bars.

    Rich ships speed columns for byte transfers, not for items, so
    this fills the gap.  Below 1/s the display inverts to seconds-
    per-item, which is easier to read at typical patent ingestion
    speeds.
    """

    def render(self, task) -> Text:
        speed = task.finished_speed or task.speed

        if not speed:
            return Text("--", style="progress.data.speed")

        if speed >= 1:
            return Text(f"{speed:.2f}/s", style="progress.data.speed")

        return Text(f"{1 / speed:.1f}s each", style="progress.data.speed")


class SmartETAColumn(ProgressColumn):
    """
    Accurate ETA column using a rolling window of recent completions.

    Rich's built-in TimeRemainingColumn derives speed from
    ``completed / elapsed`` over the entire task lifetime, which is
    inaccurate when:

    * The total changes mid-task (dynamic bars like Chunking).
    * Throughput varies (GPU warm-up, network hiccups).

    This column keeps a fixed-size deque of ``(timestamp, completed)``
    samples and computes speed from the most recent window only,
    giving a responsive and accurate ETA that reflects current
    throughput rather than a lifetime average.

    A minimum of ``min_samples`` data points is required before an
    ETA is shown — until then the column displays ``--:--``.
    """

    def __init__(
        self,
        window_size: int = 30,
        min_samples: int = 3,
        sample_interval: float = 0.5,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.min_samples = min_samples
        self.sample_interval = sample_interval

        # task_id -> deque of (timestamp, completed)
        self._samples: dict[int, collections.deque] = {}
        self._last_sample_time: dict[int, float] = {}

    def render(self, task) -> Text:
        if task.total is None or task.total == 0:
            return Text("--:--", style="progress.remaining")

        completed = task.completed
        remaining = task.total - completed

        if remaining <= 0:
            return Text("0:00:00", style="progress.remaining")

        now = time.monotonic()
        task_id = task.id

        # Initialise on first call.
        if task_id not in self._samples:
            self._samples[task_id] = collections.deque(maxlen=self.window_size)
            self._last_sample_time[task_id] = 0.0

        # Sample at most once per sample_interval to avoid flooding
        # the deque with identical values on rapid render() calls.
        if now - self._last_sample_time[task_id] >= self.sample_interval:
            self._samples[task_id].append((now, completed))
            self._last_sample_time[task_id] = now

        samples = self._samples[task_id]

        if len(samples) < self.min_samples:
            return Text("--:--", style="progress.remaining")

        # Speed = items completed across the window / time across the window.
        oldest_time, oldest_completed = samples[0]
        dt = now - oldest_time
        d_items = completed - oldest_completed

        if dt <= 0 or d_items <= 0:
            return Text("--:--", style="progress.remaining")

        speed = d_items / dt
        eta_seconds = remaining / speed

        return Text(self._format_time(eta_seconds), style="progress.remaining")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into H:MM:SS or M:SS."""
        seconds = int(seconds)
        if seconds < 0:
            return "--:--"

        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"


# ==============================================================
# IngestProgress — wraps the 5-bar display
# ==============================================================


class IngestProgress:
    """
    Manages the multi-bar Rich progress display for ingestion.

    Usage::

        progress = IngestProgress(total_files=len(txt_files))

        with progress:
            # scanning callback
            progress.advance_scanning(1)

            # when a patent's chunks are counted
            progress.advance_chunking(chunk_count)

            # embedding callback
            progress.advance_embedding(n)

            # insert callback
            progress.advance_inserting(n)

            # patent fully committed
            progress.advance_completed(1)

            # print a message above the bars
            progress.report("some message")
    """

    def __init__(self, total_files: int, quiet: bool = False) -> None:
        self._total_files = total_files
        self._lock = Lock()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description:<20}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("-"),
            RateColumn(),
            TextColumn("- elapsed"),
            TimeElapsedColumn(),
            TextColumn("- eta"),
            SmartETAColumn(),
            disable=quiet,
        )

        # Task IDs — assigned in __enter__.
        self._scan_task = None
        self._chunk_task = None
        self._embed_task = None
        self._insert_task = None
        self._complete_task = None

    # ----------------------------------------------------------
    # Context manager
    # ----------------------------------------------------------

    def __enter__(self):
        self._progress.__enter__()

        self._scan_task = self._progress.add_task(
            "Scanning Patents",
            total=self._total_files,
        )
        self._chunk_task = self._progress.add_task(
            "Chunking",
            total=None,
        )
        self._embed_task = self._progress.add_task(
            "Embedding Chunks",
            total=None,
        )
        self._insert_task = self._progress.add_task(
            "Inserting Chunks",
            total=None,
        )
        self._complete_task = self._progress.add_task(
            "Completed Patents",
            total=self._total_files,
        )

        return self

    def __exit__(self, *exc):
        return self._progress.__exit__(*exc)

    # ----------------------------------------------------------
    # Public advance helpers (all thread-safe)
    # ----------------------------------------------------------

    def advance_scanning(self, n: int = 1) -> None:
        """A patent file has been parsed + chunked by the prefetch thread."""
        with self._lock:
            self._progress.advance(self._scan_task, n)

    def advance_chunking(self, n: int) -> None:
        """
        *n* new chunks discovered for a patent.

        Also bumps the totals on the Embedding and Inserting bars so
        they know how many chunks to expect.
        """
        with self._lock:
            # Update the Chunking bar's total and advance it.
            current_chunk_total = self._progress.tasks[self._chunk_task].total or 0
            new_total = current_chunk_total + n

            self._progress.update(self._chunk_task, total=new_total)
            self._progress.advance(self._chunk_task, n)

            # Embedding and Inserting bars share the same growing total.
            self._progress.update(self._embed_task, total=new_total)
            self._progress.update(self._insert_task, total=new_total)

    def advance_embedding(self, n: int = 1) -> None:
        """Chunks have been embedded by the GPU."""
        with self._lock:
            self._progress.advance(self._embed_task, n)

    def advance_inserting(self, n: int = 1) -> None:
        """Chunks have been written to Qdrant."""
        with self._lock:
            self._progress.advance(self._insert_task, n)

    def advance_completed(self, n: int = 1) -> None:
        """A patent has been fully committed and checkpointed."""
        with self._lock:
            self._progress.advance(self._complete_task, n)

    # ----------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------

    def report(self, message: str) -> None:
        """
        Print *message* above the live progress bars.

        markup/highlight are off so patent IDs or exception messages
        containing square brackets render verbatim.
        """
        with self._lock:
            self._progress.console.print(
                message,
                markup=False,
                highlight=False,
            )

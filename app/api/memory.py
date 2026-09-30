"""
Resident memory (RSS) of this process, for comparing how much memory a search
needs per collection.

MemorySampler polls RSS on a background thread while a collection's Phases 2-7
run, so it catches the peak inside a phase, not just the value at its end. RSS
is process-wide: a search running at the same time is counted too.
"""

import ctypes
import gc
import os
import threading
from typing import Optional

MemoryUsage = dict[str, Optional[int]]

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

try:  # glibc only; elsewhere freed memory just stays with the process
    _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
except (OSError, AttributeError):
    _malloc_trim = None


def rss_bytes() -> Optional[int]:
    """Current resident set size, or None where /proc is unavailable (non-Linux)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return None


def release_free_memory() -> None:
    """
    Collect garbage and hand freed heap pages back to the OS, so a run doesn't
    look lighter just because it reuses memory an earlier run left behind.
    """
    gc.collect()
    if _malloc_trim is not None:
        _malloc_trim(0)


class MemorySampler:
    """
    Tracks RSS at the start of a run and its peak, overall and per lap (one lap
    per phase). Use as a context manager around the run; every method returns
    None when RSS can't be read.
    """

    def __init__(self, interval_s: float = 0.01) -> None:
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        rss = rss_bytes()
        self.enabled = rss is not None
        self._run_start = self._run_peak = rss
        self._lap_start = self._lap_peak = rss

    def __enter__(self) -> "MemorySampler":
        if self.enabled:
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _poll(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def _sample(self) -> Optional[int]:
        rss = rss_bytes()
        if rss is not None:
            with self._lock:
                self._lap_peak = max(self._lap_peak, rss)
                self._run_peak = max(self._run_peak, rss)
        return rss

    def end_lap(self) -> Optional[MemoryUsage]:
        """RSS when the current lap started and its peak since."""
        if not self.enabled:
            return None
        self._sample()
        with self._lock:
            return {"start_bytes": self._lap_start, "peak_bytes": self._lap_peak}

    def start_lap(self) -> None:
        rss = self._sample() if self.enabled else None
        if rss is not None:
            with self._lock:
                self._lap_start = self._lap_peak = rss

    def total(self) -> Optional[MemoryUsage]:
        """RSS when the run started, its peak, and RSS now."""
        if not self.enabled:
            return None
        end = self._sample()
        with self._lock:
            return {"start_bytes": self._run_start, "peak_bytes": self._run_peak, "end_bytes": end}

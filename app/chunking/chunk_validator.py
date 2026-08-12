"""
Chunk Validator

Validates and filters chunks before they are indexed.
Core checks: empty, whitespace-only, and duplicate.
Optional checks (all config-driven): whitespace normalization,
heading-only detection, low information detection, degenerate
overlap detection.

Does NOT reject small chunks — every section's content
is legitimate and must be preserved.
"""

from __future__ import annotations

import hashlib
import re


from app.config import (
    VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD,
    VALIDATOR_HEADING_ONLY_MAX_WORDS,
    VALIDATOR_LOW_INFO_THRESHOLD,
    VALIDATOR_NORMALIZE_WHITESPACE,
    VALIDATOR_REJECT_DEGENERATE_OVERLAP,
    VALIDATOR_REJECT_HEADING_ONLY,
    VALIDATOR_REJECT_LOW_INFO,
)


class ChunkValidator:
    """
    Stateful chunk validator.

    Maintains a set of content hashes for the current document
    to detect and reject exact duplicates.

    Call ``reset()`` between documents.
    """

    def __init__(
        self,
        *,
        normalize_whitespace: bool = VALIDATOR_NORMALIZE_WHITESPACE,
        reject_heading_only: bool = VALIDATOR_REJECT_HEADING_ONLY,
        reject_low_info: bool = VALIDATOR_REJECT_LOW_INFO,
        low_info_threshold: float = VALIDATOR_LOW_INFO_THRESHOLD,
        reject_degenerate_overlap: bool = VALIDATOR_REJECT_DEGENERATE_OVERLAP,
        degenerate_overlap_threshold: float = VALIDATOR_DEGENERATE_OVERLAP_THRESHOLD,
        heading_only_max_words: int = VALIDATOR_HEADING_ONLY_MAX_WORDS,
    ) -> None:

        # Config flags for optional checks
        self._normalize_whitespace = normalize_whitespace
        self._reject_heading_only = reject_heading_only
        self._reject_low_info = reject_low_info
        self._low_info_threshold = low_info_threshold
        self._reject_degenerate_overlap = reject_degenerate_overlap
        self._degenerate_overlap_threshold = degenerate_overlap_threshold
        self._heading_only_max_words = heading_only_max_words

        # Track content hashes for duplicate detection
        self._seen_hashes: set[str] = set()

        # Track the previous chunk text for degenerate overlap detection
        self._previous_text: str = ""

    # ==============================================================
    # Public API
    # ==============================================================

    def is_valid(self, text: str) -> bool:
        """
        Return True if *text* passes all validation checks.

        Core checks (always active):
        1. Not empty / whitespace-only
        2. Not a duplicate (by content hash)

        Optional checks (config-driven):
        3. Heading-only chunk detection
        4. Low information content detection
        5. Degenerate overlap detection

        Small chunks are NOT rejected. Every section's content
        is legitimate regardless of size.
        """

        # ---- Optional: Whitespace normalization ----
        if self._normalize_whitespace:
            text = self._normalize(text)

        stripped = text.strip()

        # ---- Check 1: empty / whitespace ----
        if not stripped:
            return False

        # ---- Check 2: duplicate ----
        content_hash = self._hash(stripped)

        if content_hash in self._seen_hashes:
            return False

        self._seen_hashes.add(content_hash)

        # ---- Check 3: heading-only ----
        if self._reject_heading_only and self._is_heading_only(stripped):
            return False

        # ---- Check 4: low information ----
        if self._reject_low_info and self._is_low_info(stripped):
            return False

        # ---- Check 5: degenerate overlap ----
        if (
            self._reject_degenerate_overlap
            and self._previous_text
            and self._is_degenerate_overlap(stripped)
        ):
            return False

        # Track for next overlap check
        self._previous_text = stripped

        return True

    def reset(self) -> None:
        """
        Clear duplicate tracking state.
        Call this between documents.
        """

        self._seen_hashes.clear()
        self._previous_text = ""

    # ==============================================================
    # Optional check: heading-only
    # ==============================================================

    def _is_heading_only(self, text: str) -> bool:
        """
        Return True if *text* looks like it contains only a section
        heading with no meaningful body content.

        Heuristic: short text (≤ max words), and is either
        ALL-CAPS or ends with a colon.
        """

        words = text.split()

        if len(words) > self._heading_only_max_words:
            return False

        # ALL-CAPS heading
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars > 0 and text == text.upper():
            return True

        # Colon-terminated heading
        if text.rstrip().endswith(":"):
            return True

        return False

    # ==============================================================
    # Optional check: low information
    # ==============================================================

    def _is_low_info(self, text: str) -> bool:
        """
        Return True if the ratio of alphabetic characters to total
        characters is below the configured threshold.

        Catches garbage like separators, pure numbering, or OCR noise.
        """

        if not text:
            return True

        total = len(text)
        alpha = sum(1 for c in text if c.isalpha())

        return (alpha / total) < self._low_info_threshold

    # ==============================================================
    # Optional check: degenerate overlap
    # ==============================================================

    def _is_degenerate_overlap(self, text: str) -> bool:
        """
        Return True if *text* is nearly identical to the previous chunk.

        Only catches truly degenerate cases where the overlap text
        dominates the chunk. Normal, intentional overlap is preserved.

        Uses word-set similarity: if the fraction of words in this
        chunk that also appear in the previous chunk exceeds the
        threshold, the chunk is degenerate.
        """

        current_words = set(text.split())
        previous_words = set(self._previous_text.split())

        if not current_words:
            return False

        shared = current_words & previous_words
        similarity = len(shared) / len(current_words)

        return similarity >= self._degenerate_overlap_threshold

    # ==============================================================
    # Internal
    # ==============================================================

    @staticmethod
    def _hash(text: str) -> str:
        """
        Fast content hash for duplicate detection.
        """

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Collapse multiple whitespace characters into single spaces.
        Prevents near-duplicates that differ only in whitespace.
        """

        return re.sub(r"\s+", " ", text).strip()

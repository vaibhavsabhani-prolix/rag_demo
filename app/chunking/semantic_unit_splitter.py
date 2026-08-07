"""
Semantic Unit Splitter

Splits section content into the smallest meaningful units:
paragraphs → sentences → word fragments (fallback).

All decisions are dynamic — no hardcoded domain words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from app.chunking.token_counter import TokenCounter
from app.config import MAX_CHUNK_TOKENS


# ==================================================================
# Semantic unit types
# ==================================================================

class UnitType(Enum):
    """Type tag for each semantic unit."""

    PARAGRAPH = auto()
    SENTENCE = auto()
    WORD_FRAGMENT = auto()


@dataclass
class SemanticUnit:
    """
    A single indivisible piece of text produced by the splitter.
    """

    text: str
    unit_type: UnitType
    token_count: int


# ==================================================================
# Generic sentence-boundary regex
# ==================================================================

# Matches a sentence-ending punctuation mark followed by whitespace,
# but avoids splitting on:
#   - Single uppercase letter + dot  (abbreviations like "U.", "A.")
#   - Common 2-3 letter abbreviations ending with dot before a space
#     (e.g. "Dr. Smith", "Fig. 1", "No. 5", "e.g. something")
#   - Decimal numbers (e.g. "3.14")
#   - Ellipsis ("...")
#
# Strategy: use a negative lookbehind to skip known non-boundary
# patterns, then require [.!?] followed by whitespace.

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!"        # negative lookbehind group start
    r"\b[A-Z]"     # single capital letter (A. B. C.)
    r")"
    r"(?<!"
    r"\b[A-Z][a-z]"  # two-letter abbreviation (Dr. Mr. Ms. Co. No.)
    r")"
    r"(?<!"
    r"\b[A-Z][a-z][a-z]"  # three-letter abbreviation (Fig. Inc. Ltd.)
    r")"
    r"(?<!"
    r"\d"          # digit before dot (3.14, 1.2)
    r")"
    r"(?<!"
    r"\.\."        # ellipsis (...)
    r")"
    r"(?<=[.!?])"  # positive lookbehind: must end with . ! ?
    r"\s+"         # consume the whitespace separator
)


# ==================================================================
# Splitter
# ==================================================================

class SemanticUnitSplitter:
    """
    Dynamic semantic unit splitter.

    Decision tree:

    1. Split content on double-newlines → paragraphs.
    2. If only one paragraph, split on sentence boundaries.
    3. If a sentence exceeds *max_tokens*, split by words into
       token-bounded fragments.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = MAX_CHUNK_TOKENS,
    ) -> None:

        self.token_counter = token_counter
        self.max_tokens = max_tokens

    # ==============================================================
    # Top-level split
    # ==============================================================

    def split(self, text: str) -> list[SemanticUnit]:
        """
        Split *text* into semantic units.
        """

        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)

        # Multiple paragraphs → each paragraph is a semantic unit
        if len(paragraphs) > 1:
            return self._process_paragraphs(paragraphs)

        # Single block → split into sentences
        return self._process_single_block(paragraphs[0])

    # ==============================================================
    # Paragraph splitting
    # ==============================================================

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """
        Split on double-newline boundaries.
        Returns non-empty paragraphs only.
        """

        raw_paragraphs = re.split(r"\n\s*\n", text)

        result: list[str] = []

        for para in raw_paragraphs:
            cleaned = para.strip()

            if cleaned:
                result.append(cleaned)

        return result if result else [text.strip()]

    # ==============================================================
    # Process multiple paragraphs
    # ==============================================================

    def _process_paragraphs(
        self,
        paragraphs: list[str],
    ) -> list[SemanticUnit]:
        """
        Convert paragraphs into semantic units.

        If a paragraph exceeds *max_tokens*, break it into sentences.
        If a sentence still exceeds, break it into word fragments.
        """

        units: list[SemanticUnit] = []

        for para in paragraphs:

            token_count = self.token_counter.count(para)

            if token_count <= self.max_tokens:
                units.append(
                    SemanticUnit(
                        text=para,
                        unit_type=UnitType.PARAGRAPH,
                        token_count=token_count,
                    )
                )
            else:
                # Paragraph too large — drill into sentences
                units.extend(self._split_sentences(para))

        return units

    # ==============================================================
    # Process a single text block (no paragraph breaks)
    # ==============================================================

    def _process_single_block(self, text: str) -> list[SemanticUnit]:
        """
        No paragraph structure → split into sentences directly.
        """

        return self._split_sentences(text)

    # ==============================================================
    # Sentence splitting
    # ==============================================================

    def _split_sentences(self, text: str) -> list[SemanticUnit]:
        """
        Split text into sentence-level semantic units.
        Oversized sentences are further split by words.
        """

        raw_sentences = _SENTENCE_SPLIT_RE.split(text)

        units: list[SemanticUnit] = []

        for sentence in raw_sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            token_count = self.token_counter.count(sentence)

            if token_count <= self.max_tokens:
                units.append(
                    SemanticUnit(
                        text=sentence,
                        unit_type=UnitType.SENTENCE,
                        token_count=token_count,
                    )
                )
            else:
                # Sentence exceeds token limit — word-level fallback
                units.extend(self._split_by_words(sentence))

        return units

    # ==============================================================
    # Word-level fallback
    # ==============================================================

    def _split_by_words(self, text: str) -> list[SemanticUnit]:
        """
        Split *text* into token-bounded word fragments.
        This is the last-resort fallback for extremely long sentences.
        """

        words = text.split()
        units: list[SemanticUnit] = []

        current_words: list[str] = []
        current_tokens = 0

        for word in words:

            word_tokens = self.token_counter.count(word)

            # +1 accounts for the space separator token between words
            projected = current_tokens + word_tokens + (1 if current_words else 0)

            if projected <= self.max_tokens:
                current_words.append(word)
                current_tokens = projected
            else:
                # Flush current fragment
                if current_words:
                    fragment_text = " ".join(current_words)
                    units.append(
                        SemanticUnit(
                            text=fragment_text,
                            unit_type=UnitType.WORD_FRAGMENT,
                            token_count=self.token_counter.count(fragment_text),
                        )
                    )

                current_words = [word]
                current_tokens = word_tokens

        # Flush remaining words
        if current_words:
            fragment_text = " ".join(current_words)
            units.append(
                SemanticUnit(
                    text=fragment_text,
                    unit_type=UnitType.WORD_FRAGMENT,
                    token_count=self.token_counter.count(fragment_text),
                )
            )

        return units

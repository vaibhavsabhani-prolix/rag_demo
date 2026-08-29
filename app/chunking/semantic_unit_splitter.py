"""
Semantic Unit Splitter

Splits section content into the smallest meaningful units:
paragraphs → sentences → word fragments → character fragments.

All decisions are dynamic — no hardcoded domain words.

Script coverage
---------------
The corpus is multilingual (English, French, Spanish, German, Chinese,
Japanese, Korean), which drives two requirements a Latin-only splitter
does not meet:

    Terminators. CJK ends sentences with the ideographic full stop and
    its fullwidth siblings, and writes no space afterwards. A rule of
    "[.!?] followed by whitespace" finds zero boundaries in Chinese or
    Japanese text, so the CJK branch below splits immediately after the
    terminator instead of requiring a following space.

    Word boundaries. Chinese and Japanese do not delimit words with
    spaces, so str.split() can return a single "word" of arbitrary
    length. Every fallback therefore bottoms out in a character-level
    split, which is the only one guaranteed to make progress on a script
    with no whitespace at all.

French, Spanish, German and Korean all terminate with [.!?] and separate
words with spaces, so they take the Latin path unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.chunking.token_counter import TokenCounter
from app.config import MAX_CHUNK_TOKENS

# ==================================================================
# Semantic unit types
# ==================================================================


@dataclass
class SemanticUnit:
    text: str
    token_count: int


# ==================================================================
# Generic sentence-boundary regex
# ==================================================================

# Closing punctuation that belongs to the sentence it follows, so a
# boundary is taken after it rather than before: He said "Stop." / 「や
# めろ。」 Splitting between the terminator and its closer would strand
# the quote at the head of the next unit.
_LATIN_CLOSERS = r"\"\'\u201d\u2019\u00bb\u203a)\]\}"
_CJK_CLOSERS = "\u300d\u300f\uff09\u3011\u3015\u300b\u201d\u2019"

# CJK sentence terminators: ideographic full stop, fullwidth exclamation
# and question marks, halfwidth ideographic full stop. The fullwidth
# full stop U+FF0E is deliberately excluded - in technical Japanese it
# is more often a decimal point than a sentence end.
_CJK_TERMINATORS = "\u3002\uff01\uff1f\uff61"

# Matches a sentence-ending boundary in either script family.
#
# This matches the terminator ITSELF (plus any closing punctuation), not
# the gap after it, for two reasons:
#
#   The negative lookbehinds below can only screen abbreviations if they
#   are evaluated immediately before the dot, where they see "Dr" and
#   "Fig". Anchored after the dot they would inspect "r." and "g." and
#   never fire.
#
#   Callers slice on match.end(), so the terminator and its closers stay
#   with the sentence they belong to. Consuming them as a separator
#   would delete them from the text outright.
#
# CJK branch: terminator plus closers, with no following space required
# - CJK does not write one, so requiring it finds nothing.
#
# Latin branch: terminator plus closers, which must be followed by
# whitespace or end-of-text, while screening out known non-boundaries:
#   - Single uppercase letter + dot  (initials like "U.", "A.")
#   - Common 2-3 letter abbreviations (Dr. Mr. Co. No. Sr. Nr. Fig.)
#   - Decimal numbers (e.g. "3.14")
#   - Ellipsis ("...")

_SENTENCE_BOUNDARY_RE = re.compile(
    # ---- CJK: boundary sits immediately after the terminator ----
    r"[" + _CJK_TERMINATORS + r"]"
    r"[" + _CJK_CLOSERS + r"]*"
    r"|"
    # ---- Latin: terminator must be followed by whitespace or end ----
    r"(?<!"  # negative lookbehind group start
    r"\b[A-Z]"  # single capital letter (A. B. C.)
    r")"
    r"(?<!"
    r"\b[A-Z][a-z]"  # two-letter abbreviation (Dr. Mr. Ms. Co. No.)
    r")"
    r"(?<!"
    r"\b[A-Z][a-z][a-z]"  # three-letter abbreviation (Fig. Inc. Ltd.)
    r")"
    r"(?<!"
    r"\d"  # digit before dot (3.14, 1.2)
    r")"
    r"(?<!"
    r"\.\."  # ellipsis (...)
    r")"
    r"[.!?\u2026]"  # the terminator itself
    r"[" + _LATIN_CLOSERS + r"]*"
    r"(?=\s|$)"  # boundary only if whitespace or end follows
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
    4. If a single "word" still exceeds *max_tokens*, split it by
        characters. Scripts without spaces reach this step for whole
        sentences at a time, and it is the only step guaranteed to
        terminate regardless of script.

    Every unit returned is at most *max_tokens*.
    """

    # A paragraph longer than max_tokens * this many characters is
    # treated as oversized without being tokenized. Real text averages
    # roughly 1-5 characters per token depending on script; 8 leaves
    # generous headroom while still short-circuiting the pathological
    # multi-megabyte case.
    CERTAINLY_OVERSIZED_CHARS_PER_TOKEN = 8

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

            # Cheap length screen first. Tokenizing a multi-megabyte
            # paragraph only to learn it is oversized costs far more
            # than the split it triggers, and no script packs anywhere
            # near CERTAINLY_OVERSIZED_CHARS_PER_TOKEN characters into a
            # token, so a paragraph past that bound is oversized without
            # counting. A false positive would merely split something
            # that fit, which the chunk builder merges back anyway.
            if len(para) > self.max_tokens * self.CERTAINLY_OVERSIZED_CHARS_PER_TOKEN:
                units.extend(self._split_sentences(para))
                continue

            token_count = self.token_counter.count(para)

            if token_count <= self.max_tokens:
                units.append(
                    SemanticUnit(
                        text=para,
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

    @staticmethod
    def _iter_sentences(text: str):
        """
        Yield sentences, cutting after each boundary match.

        Slicing on match.end() keeps the terminator and any closing
        quote or bracket attached to the sentence they end, which
        re.split() cannot do - it treats the whole match as a separator
        and drops it from the output.
        """

        start = 0

        for match in _SENTENCE_BOUNDARY_RE.finditer(text):
            yield text[start:match.end()]
            start = match.end()

        if start < len(text):
            yield text[start:]

    def _split_sentences(self, text: str) -> list[SemanticUnit]:
        """
        Split text into sentence-level semantic units.
        Oversized sentences are further split by words.
        """

        units: list[SemanticUnit] = []

        for sentence in self._iter_sentences(text):

            sentence = sentence.strip()

            if not sentence:
                continue

            token_count = self.token_counter.count(sentence)

            if token_count <= self.max_tokens:
                units.append(
                    SemanticUnit(
                        text=sentence,
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

        Fallback for sentences that exceed *max_tokens*. A word that on
        its own exceeds the limit cannot be packed into any fragment, so
        it is handed to the character-level split rather than emitted
        oversized - without that escape hatch, a script that writes no
        spaces yields one "word" for the whole sentence and this method
        returns it unchanged, however long it is.
        """

        words = text.split()
        units: list[SemanticUnit] = []

        current_words: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current_words, current_tokens

            if not current_words:
                return

            fragment_text = " ".join(current_words)
            actual_tokens = self.token_counter.count(fragment_text)

            # Packing above budgets with per-word counts plus one token
            # per separator, but the joined text can tokenize to
            # slightly more than the sum of its parts - subword merges
            # differ across a boundary. Re-measure and fall through to
            # the character split when the estimate came up short, so
            # the max_tokens guarantee holds on the real text.
            if actual_tokens > self.max_tokens:
                units.extend(self._split_by_characters(fragment_text))
            else:
                units.append(
                    SemanticUnit(
                        text=fragment_text,
                        token_count=actual_tokens,
                    )
                )

            current_words = []
            current_tokens = 0

        for word in words:

            word_tokens = self.token_counter.count(word)

            # No fragment can ever hold this word - split the word
            # itself, keeping it in order relative to its neighbours.
            if word_tokens > self.max_tokens:
                flush()
                units.extend(self._split_by_characters(word))
                continue

            # +1 accounts for the space separator token between words
            projected = current_tokens + word_tokens + (1 if current_words else 0)

            if projected <= self.max_tokens:
                current_words.append(word)
                current_tokens = projected
            else:
                flush()

                current_words = [word]
                current_tokens = word_tokens

        flush()

        return units

    # ==============================================================
    # Character-level fallback
    # ==============================================================

    def _split_by_characters(self, text: str) -> list[SemanticUnit]:
        """
        Split *text* into token-bounded fragments on character offsets.

        The terminal fallback. Chinese and Japanese write no spaces, so
        for those scripts this - not the word split - is what actually
        bounds a run of text, and it is the only split that makes
        progress on any input whatsoever.

        Tokenizing one character at a time would be far too slow at
        patent scale, so each step estimates a window from the observed
        characters-per-token ratio, then shrinks it until it fits. The
        estimate is derived from this text, so it adapts to the script
        rather than assuming one.
        """

        if not text:
            return []

        total_tokens = self.token_counter.count(text)

        if total_tokens <= self.max_tokens:
            return [SemanticUnit(text=text, token_count=total_tokens)]

        chars_per_token = len(text) / total_tokens
        window = max(int(self.max_tokens * chars_per_token), 1)

        units: list[SemanticUnit] = []
        start = 0

        while start < len(text):

            piece = text[start:start + window]
            token_count = self.token_counter.count(piece)

            # Shrink proportionally until the piece fits. Guaranteed to
            # terminate: each pass strictly shortens the piece, and a
            # single character always ends the loop.
            while token_count > self.max_tokens and len(piece) > 1:
                shrunk = int(len(piece) * self.max_tokens / token_count)
                piece = text[start:start + max(min(shrunk, len(piece) - 1), 1)]
                token_count = self.token_counter.count(piece)

            units.append(
                SemanticUnit(
                    text=piece,
                    token_count=token_count,
                )
            )

            start += len(piece)

        return units

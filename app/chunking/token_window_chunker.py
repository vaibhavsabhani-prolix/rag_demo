"""
Token Window Chunker

Fast, token-aware chunker with local boundary adjustment.

Replaces the hierarchical split-and-merge approach by:
1. Tokenizing the section content ONCE to obtain token IDs and character offsets.
2. Creating target token windows bounded by MAX_CHUNK_TOKENS.
3. Adjusting boundaries locally backward to the nearest safe textual boundary:
   - Priority 1: Paragraph boundary (\\n\\s*\\n)
   - Priority 2: Sentence boundary (.!?, CJK terminators, abbreviations, decimals)
   - Priority 3: Word boundary (whitespace)
   - Priority 4: Character boundary (clean UTF-8 token boundary fallback)
4. Slicing original section text using tokenizer character offsets.
5. Emitting chunks with exact token counts (end_tok - start_tok) without re-tokenization.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import re

from app.chunking.token_counter import TokenCounter
from app.config import MAX_CHUNK_TOKENS


@dataclass
class BuiltChunk:
    """
    A chunk produced by the chunker before final validation.
    """
    text: str
    section: str
    token_count: int
    word_count: int


# ==================================================================
# Generic sentence-boundary regex (Multilingual: Latin + CJK)
# ==================================================================

# Closing punctuation that belongs to the sentence it follows:
_LATIN_CLOSERS = r"\"\'\u201d\u2019\u00bb\u203a)\]\}"
_CJK_CLOSERS = "\u300d\u300f\uff09\u3011\u3015\u300b\u201d\u2019"

# CJK sentence terminators: ideographic full stop, fullwidth exclamation
# and question marks, halfwidth ideographic full stop.
_CJK_TERMINATORS = "\u3002\uff01\uff1f\uff61"

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


class TokenWindowChunker:
    """
    Token-window chunker with multi-tier boundary adjustment.

    Guarantees:
    - Every produced chunk satisfies token_count <= max_tokens.
    - Boundaries never cross section boundaries.
    - Zero redundant tokenizer invocations during chunk creation.
    - Multilingual safety: handles Latin and CJK scripts, abbreviations,
      decimal numbers, and unbounded-length words.
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        max_tokens: int = MAX_CHUNK_TOKENS,
        search_window_ratio: float = 0.25,
    ) -> None:
        self.token_counter = token_counter or TokenCounter()
        self.max_tokens = max_tokens
        self.search_window_ratio = search_window_ratio

    def chunk(
        self,
        section_heading: str,
        content: str,
    ) -> list[BuiltChunk]:
        """
        Split section *content* into token-bounded chunks with boundary adjustment.

        Tokenizes *content* exactly once, then slices along adjusted safe
        boundaries without re-invoking the tokenizer.
        """
        if not content or not content.strip():
            return []

        # ---- Step 1: Tokenize section once ----
        input_ids, offsets = self.token_counter.tokenize_with_offsets(content)
        total_tokens = len(input_ids)

        if total_tokens == 0:
            return []

        # ---- Precompute boundary token indices ----
        end_positions = [e for _, e in offsets]

        # Priority 1 boundaries: Paragraph breaks (\n\s*\n)
        para_token_boundaries: list[int] = []
        for match in re.finditer(r"\n\s*\n", content):
            idx = bisect.bisect_right(end_positions, match.end()) - 1
            if idx >= 0:
                para_token_boundaries.append(idx + 1)
        para_token_boundaries = sorted(set(para_token_boundaries))

        # Priority 2 boundaries: Sentence ends (Latin and CJK)
        sent_token_boundaries: list[int] = []
        for match in _SENTENCE_BOUNDARY_RE.finditer(content):
            idx = bisect.bisect_right(end_positions, match.end()) - 1
            if idx >= 0:
                sent_token_boundaries.append(idx + 1)
        sent_token_boundaries = sorted(set(sent_token_boundaries))

        chunks: list[BuiltChunk] = []
        start_tok = 0
        lookback = max(int(self.max_tokens * self.search_window_ratio), min(50, self.max_tokens))

        # ---- Step 2 & 3: Window iteration and boundary adjustment ----
        while start_tok < total_tokens:
            remaining_tokens = total_tokens - start_tok

            # Entire remainder fits in one chunk
            if remaining_tokens <= self.max_tokens:
                end_tok = total_tokens
                char_start = offsets[start_tok][0]
                char_end = offsets[end_tok - 1][1]
                chunk_text = content[char_start:char_end].strip()

                if chunk_text:
                    chunks.append(
                        BuiltChunk(
                            text=chunk_text,
                            section=section_heading,
                            token_count=end_tok - start_tok,
                            word_count=len(chunk_text.split()),
                        )
                    )
                break

            target_tok = start_tok + self.max_tokens
            min_tok = max(start_tok + 1, target_tok - lookback)

            chosen_tok: int | None = None

            # Priority 1: Paragraph boundary within lookback window
            p_idx = bisect.bisect_right(para_token_boundaries, target_tok) - 1
            if p_idx >= 0 and para_token_boundaries[p_idx] >= min_tok:
                cand = para_token_boundaries[p_idx]
                if start_tok < cand <= target_tok:
                    chosen_tok = cand

            # Priority 2: Sentence boundary within lookback window
            if chosen_tok is None:
                s_idx = bisect.bisect_right(sent_token_boundaries, target_tok) - 1
                if s_idx >= 0 and sent_token_boundaries[s_idx] >= min_tok:
                    cand = sent_token_boundaries[s_idx]
                    if start_tok < cand <= target_tok:
                        chosen_tok = cand

            # Priority 3: Word boundary (whitespace)
            if chosen_tok is None:
                # Search backward from target_tok for whitespace boundary
                word_search_limit = max(start_tok + 1, target_tok - lookback)
                for t in range(target_tok, word_search_limit - 1, -1):
                    if t < total_tokens:
                        next_s = offsets[t][0]
                        if next_s > 0 and (
                            content[next_s - 1].isspace()
                            or (next_s < len(content) and content[next_s].isspace())
                        ):
                            chosen_tok = t
                            break

            # Priority 4: Character boundary fallback (guaranteed progress)
            if chosen_tok is None:
                chosen_tok = target_tok
                # Avoid cutting across multiple tokens that represent a single UTF-8 char
                while chosen_tok > start_tok + 1 and offsets[chosen_tok - 1][1] == offsets[chosen_tok][1]:
                    chosen_tok -= 1

            # Slicing directly from original content using token offsets
            char_start = offsets[start_tok][0]
            char_end = offsets[chosen_tok - 1][1]
            chunk_text = content[char_start:char_end].strip()

            if chunk_text:
                chunks.append(
                    BuiltChunk(
                        text=chunk_text,
                        section=section_heading,
                        token_count=chosen_tok - start_tok,
                        word_count=len(chunk_text.split()),
                    )
                )

            start_tok = chosen_tok

        return chunks


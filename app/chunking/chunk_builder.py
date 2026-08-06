"""
Chunk Builder

Token-aware greedy merging of semantic units into chunks.
Uses semantic overlap (last sentence / last unit) instead of
character-based slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking.token_counter import TokenCounter
from app.chunking.semantic_unit_splitter import SemanticUnit, UnitType
from app.config import MAX_CHUNK_TOKENS, OVERLAP_STRATEGY


# ==================================================================
# Built chunk
# ==================================================================

@dataclass
class BuiltChunk:
    """
    A chunk produced by the builder before final validation.
    """

    text: str
    section: str
    token_count: int
    word_count: int


# ==================================================================
# Builder
# ==================================================================

class ChunkBuilder:
    """
    Token-aware greedy chunk builder with semantic overlap.

    Algorithm:

    1. For each semantic unit, check whether adding it to the
       current chunk would exceed *max_tokens*.
    2. If it fits, append.
    3. If it does not fit, finalize the current chunk, extract
       semantic overlap from it, and start a new chunk
       with the overlap prepended.

    Semantic overlap modes:
        - ``"last_sentence"`` — last sentence-level unit
        - ``"last_unit"``     — last unit regardless of type
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = MAX_CHUNK_TOKENS,
        overlap_strategy: str = OVERLAP_STRATEGY,
    ) -> None:

        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self.overlap_strategy = overlap_strategy

    # ==============================================================
    # Public API
    # ==============================================================

    def build(
        self,
        section_heading: str,
        units: list[SemanticUnit],
    ) -> list[BuiltChunk]:
        """
        Merge *units* into token-bounded chunks for a single section.

        The section heading is stored as metadata in BuiltChunk.section.
        It is NOT included in the chunk text so that the embedding
        model only sees the actual content.
        """

        if not units:
            return []

        chunks: list[BuiltChunk] = []

        # ---- State ----
        current_texts: list[str] = []
        current_units: list[SemanticUnit] = []
        current_tokens: int = 0
        overlap_text: str = ""
        overlap_tokens: int = 0

        for unit in units:

            # How many tokens this unit would add
            # +1 for the newline/space separator
            separator_tokens = 1 if current_texts else 0
            projected = current_tokens + unit.token_count + separator_tokens

            # ---- Unit fits into current chunk ----
            if projected <= self.max_tokens:
                current_texts.append(unit.text)
                current_units.append(unit)
                current_tokens = projected
                continue

            # ---- Unit does NOT fit — finalize current chunk ----
            if current_texts:
                chunk = self._finalize_chunk(
                    texts=current_texts,
                    section=section_heading,
                    accumulated_tokens=current_tokens,
                )
                chunks.append(chunk)

                # Extract semantic overlap from the chunk we just closed
                overlap_text, overlap_tokens = self._extract_overlap(
                    current_units,
                )

            # ---- Start new chunk with overlap ----
            current_texts = []
            current_units = []
            current_tokens = 0

            # Prepend overlap only if overlap + unit still fits
            if overlap_text:
                projected_with_overlap = (
                    overlap_tokens + 1          # overlap + separator
                    + unit.token_count + 1      # unit + separator
                )

                if projected_with_overlap <= self.max_tokens:
                    current_texts.append(overlap_text)
                    current_tokens += overlap_tokens + 1

            # Add the current unit
            separator_tokens = 1 if current_texts else 0
            current_texts.append(unit.text)
            current_units.append(unit)
            current_tokens += unit.token_count + separator_tokens

        # ---- Flush remaining ----
        if current_texts:
            chunk = self._finalize_chunk(
                texts=current_texts,
                section=section_heading,
                accumulated_tokens=current_tokens,
            )
            chunks.append(chunk)

        return chunks

    # ==============================================================
    # Finalize a chunk
    # ==============================================================

    def _finalize_chunk(
        self,
        texts: list[str],
        section: str,
        accumulated_tokens: int,
    ) -> BuiltChunk:
        """
        Join text fragments and compute metadata.

        The section heading is NOT included in the text.
        It is stored separately in BuiltChunk.section so that
        the embedding model only encodes the actual content.

        Uses the pre-computed *accumulated_tokens* instead of
        re-tokenizing the joined body — the accumulator already
        accounts for separator tokens.
        """

        body = "\n\n".join(texts)

        return BuiltChunk(
            text=body.strip(),
            section=section,
            token_count=accumulated_tokens,
            word_count=len(body.split()),
        )

    # ==============================================================
    # Semantic overlap extraction
    # ==============================================================

    def _extract_overlap(
        self,
        units: list[SemanticUnit],
    ) -> tuple[str, int]:
        """
        Extract overlap content from the given units.

        Returns ``(overlap_text, overlap_tokens)``.
        Returns ``("", 0)`` if no suitable overlap unit exists.
        """

        if not units:
            return "", 0

        if self.overlap_strategy == "last_sentence":
            return self._overlap_last_sentence(units)

        if self.overlap_strategy == "last_unit":
            return self._overlap_last_unit(units)

        # Fallback: no overlap
        return "", 0

    @staticmethod
    def _overlap_last_sentence(
        units: list[SemanticUnit],
    ) -> tuple[str, int]:
        """
        Return the last SENTENCE-type unit as overlap.
        If no sentence exists, fall back to the last unit.
        """

        # Search backwards for a sentence
        for unit in reversed(units):

            if unit.unit_type == UnitType.SENTENCE:
                return unit.text, unit.token_count

        # Fallback to last unit of any type
        last = units[-1]
        return last.text, last.token_count

    @staticmethod
    def _overlap_last_unit(
        units: list[SemanticUnit],
    ) -> tuple[str, int]:
        """
        Return the last semantic unit regardless of type.
        """

        last = units[-1]
        return last.text, last.token_count

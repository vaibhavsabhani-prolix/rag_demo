"""
Chunk Builder

Token-aware greedy merging of semantic units into chunks.
Chunks are fully independent — no sentence or unit is ever
duplicated across chunk boundaries. Context beyond a chunk's
own bounds is reconstructed at retrieval time from neighboring
chunks, not by storing duplicated overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking.semantic_unit_splitter import SemanticUnit
from app.config import MAX_CHUNK_TOKENS

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
    Token-aware greedy chunk builder.

    Algorithm:

    1. For each semantic unit, check whether adding it to the
       current chunk would exceed *max_tokens*.
    2. If it fits, append.
    3. If it does not fit, finalize the current chunk and start
       a new chunk with the current unit.
    """

    def __init__(
        self,
        max_tokens: int = MAX_CHUNK_TOKENS,
    ) -> None:

        self.max_tokens = max_tokens

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
        current_tokens: int = 0

        for unit in units:

            # How many tokens this unit would add
            # +1 for the newline/space separator
            separator_tokens = 1 if current_texts else 0
            projected = current_tokens + unit.token_count + separator_tokens

            # ---- Unit fits into current chunk ----
            if projected <= self.max_tokens:
                current_texts.append(unit.text)
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

            # ---- Start new chunk with the current unit ----
            current_texts = [unit.text]
            current_tokens = unit.token_count

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

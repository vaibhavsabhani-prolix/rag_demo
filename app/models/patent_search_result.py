"""
Patent Search Result Model

Represents a single patent as a retrieval unit, aggregated from
multiple reranked chunk results.

The internal retrieval unit is still a Chunk.
The external retrieval unit presented to the user is a Patent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankedChunk:
    """
    A single chunk with its reranker score attached.

    Used internally by PatentSearchResult to preserve
    per-chunk scoring for downstream use (answer generation,
    citation, neighboring chunk retrieval).
    """

    chunk_id: int
    section: str
    text: str
    score: float
    token_count: int = 0
    word_count: int = 0
    section_chunk_index: int = 0
    document_chunk_index: int = 0
    total_chunks: int = 0


@dataclass
class PatentSearchResult:
    """
    One patent as a search result.

    Aggregates all matching chunks from a single patent
    after reranking, scored by the best chunk's reranker score.

    Attributes:
        patent_id:        Unique patent identifier.
        score:            Patent-level score (max reranker score).
        best_chunk:       The chunk with the highest reranker score.
        matching_chunks:  All matching chunks from this patent,
                          sorted by reranker score (descending).
        metadata:         Patent metadata from the original document.
    """

    patent_id: str
    score: float
    best_chunk: RankedChunk
    matching_chunks: list[RankedChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """Number of matching chunks from this patent."""
        return len(self.matching_chunks)

    @property
    def sections(self) -> list[str]:
        """Unique sections represented in matching chunks."""
        seen = []
        for chunk in self.matching_chunks:
            if chunk.section not in seen:
                seen.append(chunk.section)
        return seen

    @property
    def preview(self) -> str:
        """Preview text from the best matching chunk."""
        return self.best_chunk.text

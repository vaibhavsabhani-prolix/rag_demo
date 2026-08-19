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
class ScoreBreakdown:
    """
    Per-signal reranker scoring detail for one chunk - a byproduct of
    Reranker._blend_and_sort's blend that already exists in memory
    while scoring, kept here for debugging/tuning rather than
    discarded. Always populated when a chunk goes through the
    reranker (see app/reranker.py), including the pure-semantic
    fallback path (where every non-semantic field just stays at its
    neutral default and final_score == semantic_score).

    All model scores are relevance scores rather than statistical
    probabilities. Local CrossEncoder scores are sigmoid-bounded to
    the [0,1] range; remote scores are server-provided relevance
    scores following the configured endpoint's score contract.
    Sigmoid bounding does not imply statistical calibration.

    Blend weights are bounded configuration-derived values.

    structure_coverage: how well this chunk covers the query's
    already-extracted structure (mean of structured_score/
    relationship_score, whichever are present - see
    Reranker._blend_and_sort), used as a penalty-only multiplier on
    final_score. Stays at its neutral default (1.0, i.e. no penalty
    applied) when the query has no structured sentence/relationships
    to check coverage against, or for non-fine-stage chunks.
    """

    semantic_score: float = 0.0
    structured_score: float = 0.0
    relationship_score: float = 0.0
    optimization_score: float = 0.0
    lexical_score: float = 0.0
    exact_match: float = 0.0
    exclusion_penalty: float = 0.0
    structure_coverage: float = 1.0
    final_score: float = 0.0
    weights_used: dict = field(default_factory=dict)


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
    breakdown: ScoreBreakdown | None = None


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

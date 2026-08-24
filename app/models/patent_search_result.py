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
    Per-signal reranker scoring detail for one chunk.
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
    question_score: float = 0.0
    answer_relevance_score: float = 0.0
    request_satisfaction_score: float = 0.0
    request_satisfaction_label: str = "UNASSESSED"
    request_satisfaction_reason: str = ""


@dataclass
class AnswerEvidence:
    """
    Structured answer and exact supporting evidence from the original patent chunk text.
    start_char and end_char MUST reference exact character indices into chunk.text.
    """

    answer: str
    evidence_text: str
    start_char: int
    end_char: int
    confidence: float | None = None


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
    answer: str | None = None
    answer_evidence: list[AnswerEvidence] = field(default_factory=list)
    answer_span: tuple[int, int] | None = None
    answer_score: float | None = None
    highlighted_text: str | None = None
    answer_debug: dict | None = None


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
        answer:           Optional concise extracted answer for question queries.
        answer_evidence:  Optional list of AnswerEvidence supporting the answer.
        answer_span:      Optional primary character span (start, end) in chunk text.
        answer_score:     Optional confidence score that this patent answers the question.
        highlighted_text: Optional chunk text with highlighted answer evidence spans.
    """

    patent_id: str
    score: float
    best_chunk: RankedChunk
    matching_chunks: list[RankedChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    answer: str | None = None
    answer_evidence: list[AnswerEvidence] = field(default_factory=list)
    answer_span: tuple[int, int] | None = None
    answer_score: float | None = None
    highlighted_text: str | None = None
    answer_debug: dict | None = None

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

    @property
    def request_satisfaction_score(self) -> float:
        """Patent-level request satisfaction score from its best representative chunk."""
        if self.best_chunk and self.best_chunk.breakdown:
            return self.best_chunk.breakdown.request_satisfaction_score
        return 0.0

    @property
    def request_satisfaction_label(self) -> str:
        """Patent-level request satisfaction label ('DIRECT_MATCH', 'PARTIAL_MATCH', 'NON_MATCH', 'UNASSESSED')."""
        if self.best_chunk and self.best_chunk.breakdown:
            return self.best_chunk.breakdown.request_satisfaction_label
        return "UNASSESSED"

    @property
    def request_satisfaction_reason(self) -> str:
        """Patent-level request satisfaction reason from its best representative chunk."""
        if self.best_chunk and self.best_chunk.breakdown:
            return self.best_chunk.breakdown.request_satisfaction_reason
        return ""

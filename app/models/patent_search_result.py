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
    A single chunk with its patent's reranker score attached.

    Used internally by PatentSearchResult to preserve
    per-chunk scoring for downstream use (answer generation,
    citation, neighboring chunk retrieval).

    `score` is the patent-level relevance score (0-10): the MAX of that
    patent's own chunks' individual scores (see Reranker.rerank - every
    one of a patent's chunks is checked, not just a similarity-biased
    subset), so every chunk of the same patent shares this same value.
    `chunk_relevance_score` is THIS chunk's own individual score - used
    to identify which chunk actually earned the patent's max score, so
    it can be chosen as the patent's display representative.
    """

    chunk_id: int
    section: str
    text: str
    score: float
    chunk_relevance_score: float = 0.0
    token_count: int = 0
    word_count: int = 0
    section_chunk_index: int = 0
    document_chunk_index: int = 0
    total_chunks: int = 0
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

    Aggregates all matching chunks from a single patent after
    reranking. A patent's score is the MAX of its own chunks'
    individual relevance scores (see Reranker.rerank - every chunk the
    patent has is checked), so every chunk of the same patent shares
    this same score.

    Attributes:
        patent_id:        Unique patent identifier.
        score:            Patent-level relevance score, 0-10 (the max
                          across this patent's own chunks).
        best_chunk:       The specific chunk that earned that max score
                          - see RankedChunk.chunk_relevance_score.
        matching_chunks:  All matching chunks from this patent,
                          sorted by chunk_relevance_score (descending).
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
    def has_answer(self) -> bool:
        """True when a question-query answer was extracted for this patent."""
        return bool(self.answer)

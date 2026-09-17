"""
Semantic Search Pipeline — Orchestration Layer

Threads a raw user query through all seven phases in order, handing each
phase's typed result to the next:

    Phase 1  QueryUnderstandingEngine.parse            -> ParsedQuery
    Phase 2  CandidateRetriever.retrieve_candidates     -> CandidateRetrievalResult
    Phase 3  filter_candidates                          -> FilteredCandidateResult
    Phase 4  EvidenceRetriever.retrieve_evidence         -> EvidenceRetrievalResult
    Phase 5  RelationshipVerifier.verify_candidates      -> VerificationBatchResult
    Phase 6  BGEReranker.rerank_candidates               -> RerankBatchResult
    Phase 7  FinalScorer.score_and_rank                  -> FinalSearchResult

Every phase-specific decision (thresholds, batching, prompt building, etc.)
lives in that phase's own module — this file only sequences them and reports
timing/progress via an optional callback, so callers (e.g. the Streamlit UI)
can render each phase's result as soon as it is ready.
"""

import time
from typing import Callable, Optional

from app.models.scoring import FinalSearchResult
from app.query_understanding.engine import QueryUnderstandingEngine
from app.reranking.reranker import BGEReranker
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.metadata_filter import filter_candidates
from app.retrieval.retriever import CandidateRetriever
from app.scoring.scorer import FinalScorer
from app.verification.verifier import RelationshipVerifier

# Called after each phase completes: (phase_num, phase_name, result, elapsed_ms).
PhaseCallback = Callable[[int, str, object, float], None]

PHASE_NAMES = {
    1: "Query Understanding",
    2: "Candidate Vector Retrieval",
    3: "Metadata Filtering",
    4: "Bounded Evidence Retrieval",
    5: "Semantic Relationship Verification",
    6: "BGE Cross-Encoder Reranking",
    7: "Final Patent Scoring",
}


class SearchPipeline:
    """
    Orchestrates the full 7-phase patent search pipeline end to end.
    """

    def __init__(
        self,
        engine: QueryUnderstandingEngine,
        retriever: CandidateRetriever,
        evidence_retriever: EvidenceRetriever,
        verifier: RelationshipVerifier,
        reranker: BGEReranker,
        scorer: FinalScorer,
    ):
        self.engine = engine
        self.retriever = retriever
        self.evidence_retriever = evidence_retriever
        self.verifier = verifier
        self.reranker = reranker
        self.scorer = scorer

    def _emit(
        self,
        on_phase_complete: Optional[PhaseCallback],
        phase_num: int,
        result: object,
        elapsed_ms: float,
    ) -> None:
        if on_phase_complete is not None:
            on_phase_complete(phase_num, PHASE_NAMES[phase_num], result, elapsed_ms)

    def run(
        self,
        query: str,
        use_cache: bool = True,
        on_phase_complete: Optional[PhaseCallback] = None,
    ) -> FinalSearchResult:
        """
        Run Phase 1 through Phase 7 in order for *query*, invoking
        *on_phase_complete* after each phase finishes. Returns Phase 7's
        FinalSearchResult.
        """

        # Phase 1 — Query Understanding
        t0 = time.perf_counter()
        parsed_query = self.engine.parse(query, use_cache=use_cache)
        self._emit(on_phase_complete, 1, parsed_query, (time.perf_counter() - t0) * 1000)

        # Phase 2 — Candidate Retrieval (Vector Search)
        t0 = time.perf_counter()
        retrieval_result = self.retriever.retrieve_candidates(parsed_query)
        self._emit(on_phase_complete, 2, retrieval_result, (time.perf_counter() - t0) * 1000)

        # Phase 3 — Metadata Filtering & Constraint Enforcement
        t0 = time.perf_counter()
        filtered_result = filter_candidates(
            candidates=retrieval_result.candidates,
            metadata_filters=parsed_query.metadata_filters,
            is_metadata_only=parsed_query.is_metadata_only,
        )
        self._emit(on_phase_complete, 3, filtered_result, (time.perf_counter() - t0) * 1000)

        # Phase 4 — Bounded Evidence Retrieval
        t0 = time.perf_counter()
        evidence_result = self.evidence_retriever.retrieve_evidence(
            parsed_query, filtered_result.candidates
        )
        self._emit(on_phase_complete, 4, evidence_result, (time.perf_counter() - t0) * 1000)

        # Phase 5 — Semantic Relationship & Requirement Verification
        t0 = time.perf_counter()
        verification_result = self.verifier.verify_candidates(parsed_query, evidence_result)
        self._emit(on_phase_complete, 5, verification_result, (time.perf_counter() - t0) * 1000)

        # Phase 6 — BGE Cross-Encoder Reranking
        t0 = time.perf_counter()
        rerank_result = self.reranker.rerank_candidates(
            parsed_query, verification_result, evidence_result
        )
        self._emit(on_phase_complete, 6, rerank_result, (time.perf_counter() - t0) * 1000)

        # Phase 7 — Final Patent Scoring & Result Selection
        t0 = time.perf_counter()
        final_result = self.scorer.score_and_rank(rerank_result)
        self._emit(on_phase_complete, 7, final_result, (time.perf_counter() - t0) * 1000)

        print(f"[FinalResult] query={query[:80]!r} -> {len(final_result.results)} qualifying patent(s)")
        for r in final_result.results:
            print(f"[FinalResult]   {r.patent_id}  score={r.final_score:.2f}")

        return final_result

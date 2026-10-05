"""
Semantic Search Pipeline — Orchestration Layer

Threads a raw user query through all six phases in order, handing each
phase's typed result to the next:

    Phase 1  QueryUnderstandingEngine.parse            -> ParsedQuery
    Phase 2  CandidateRetriever.retrieve_candidates     -> CandidateRetrievalResult
    Phase 3  filter_candidates                          -> FilteredCandidateResult
    Phase 4  RelationshipVerifier.verify_candidates      -> VerificationBatchResult
    Phase 5  BGEReranker.rerank_candidates               -> RerankBatchResult
    Phase 6  FinalScorer.score_and_rank                  -> FinalSearchResult

Phase 4 and 5 consume evidence chunks straight from Phase 2/3's candidate
chunks (`_candidates_to_evidence_result`) — there is no separate evidence
retrieval phase; Phase 2 already fetched the chunk text these phases need,
so no new Qdrant calls happen between Phase 3 and Phase 4.

Every phase-specific decision (thresholds, batching, prompt building, etc.)
lives in that phase's own module — this file only sequences them and reports
timing/progress via an optional callback, so callers (e.g. the Streamlit UI)
can render each phase's result as soon as it is ready.
"""

import time
from typing import Callable, List, Optional

from app.models.candidate import CandidatePatent
from app.models.collection import SearchCollection
from app.models.evidence import EvidenceChunk, EvidenceRetrievalResult, PatentEvidence
from app.models.parsed_query import ParsedQuery
from app.models.scoring import FinalSearchResult
from app.query_understanding.engine import QueryUnderstandingEngine
from app.reranking.reranker import BGEReranker
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
    4: "Semantic Relationship Verification",
    5: "BGE Cross-Encoder Reranking",
    6: "Final Patent Scoring",
}


def _candidates_to_evidence_result(candidates: List[CandidatePatent]) -> EvidenceRetrievalResult:
    """
    Build the evidence structure Phase 4/5 consume directly from Phase 2/3's
    candidate chunks — no new retrieval, just a reshape of data already fetched.
    """
    evidence_by_patent: dict[str, List[EvidenceChunk]] = {}
    patent_evidence_list: List[PatentEvidence] = []
    total_chunks = 0

    for cand in candidates:
        chunks = [
            EvidenceChunk(
                patent_id=cand.patent_id,
                chunk_id=c.chunk_id,
                text=c.text or "",
                retrieval_score=c.score,
                retrieval_source="candidate_retrieval",
                section=c.section,
                document_chunk_index=c.document_chunk_index,
                token_count=c.token_count,
            )
            for c in cand.chunks
        ]
        evidence_by_patent[cand.patent_id] = chunks
        patent_evidence_list.append(
            PatentEvidence(
                patent_id=cand.patent_id,
                chunks=chunks,
                metadata=cand.metadata,
                candidate_score=cand.retrieval_score,
            )
        )
        total_chunks += len(chunks)

    return EvidenceRetrievalResult(
        evidence_by_patent=evidence_by_patent,
        patent_evidence_list=patent_evidence_list,
        total_candidates=len(candidates),
        total_evidence_chunks=total_chunks,
    )


class SearchPipeline:
    """
    Orchestrates the full 6-phase patent search pipeline end to end.
    """

    def __init__(
        self,
        engine: QueryUnderstandingEngine,
        retriever: CandidateRetriever,
        verifier: RelationshipVerifier,
        reranker: BGEReranker,
        scorer: FinalScorer,
    ):
        self.engine = engine
        self.retriever = retriever
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
        collection: SearchCollection,
        use_cache: bool = True,
        on_phase_complete: Optional[PhaseCallback] = None,
    ) -> FinalSearchResult:
        """
        Run Phase 1 through Phase 6 in order for *query* against the Qdrant
        *collection*, invoking *on_phase_complete* after each phase
        finishes. Returns Phase 6's FinalSearchResult.
        """

        # Phase 1 — Query Understanding
        t0 = time.perf_counter()
        parsed_query = self.engine.parse(query, use_cache=use_cache)
        self._emit(on_phase_complete, 1, parsed_query, (time.perf_counter() - t0) * 1000)

        return self.run_parsed(parsed_query, collection, on_phase_complete)

    def run_parsed(
        self,
        parsed_query: ParsedQuery,
        collection: SearchCollection,
        on_phase_complete: Optional[PhaseCallback] = None,
    ) -> FinalSearchResult:
        """
        Run Phase 2 through Phase 6 for an already parsed query. Phase 1 does
        not depend on the collection, so comparing collections parses once
        and calls this per collection.
        """

        # Phase 2 — Candidate Retrieval (Vector Search)
        t0 = time.perf_counter()
        retrieval_result = self.retriever.retrieve_candidates(parsed_query, collection)
        self._emit(on_phase_complete, 2, retrieval_result, (time.perf_counter() - t0) * 1000)

        # Phase 3 — Metadata Filtering & Constraint Enforcement
        t0 = time.perf_counter()
        filtered_result = filter_candidates(
            candidates=retrieval_result.candidates,
            metadata_filters=parsed_query.metadata_filters,
            is_metadata_only=parsed_query.is_metadata_only,
        )
        self._emit(on_phase_complete, 3, filtered_result, (time.perf_counter() - t0) * 1000)

        # Evidence for Phase 4/5 comes straight from Phase 2/3's candidate chunks —
        # no new retrieval, so this isn't its own phase/callback.
        evidence_result = _candidates_to_evidence_result(filtered_result.candidates)

        # Phase 4 — Semantic Relationship & Requirement Verification
        t0 = time.perf_counter()
        verification_result = self.verifier.verify_candidates(parsed_query, evidence_result)
        self._emit(on_phase_complete, 4, verification_result, (time.perf_counter() - t0) * 1000)

        # Phase 5 — BGE Cross-Encoder Reranking
        t0 = time.perf_counter()
        rerank_result = self.reranker.rerank_candidates(
            parsed_query, verification_result, evidence_result
        )
        self._emit(on_phase_complete, 5, rerank_result, (time.perf_counter() - t0) * 1000)

        # Phase 6 — Final Patent Scoring & Result Selection
        t0 = time.perf_counter()
        final_result = self.scorer.score_and_rank(
            rerank_result, is_metadata_only=parsed_query.is_metadata_only
        )
        self._emit(on_phase_complete, 6, final_result, (time.perf_counter() - t0) * 1000)

        print(
            f"[FinalResult] query={parsed_query.original_query[:80]!r} "
            f"collection={collection.name} -> {len(final_result.results)} qualifying patent(s)"
        )
        for r in final_result.results:
            score = "unscored" if r.final_score is None else f"{r.final_score:.2f}"
            print(f"[FinalResult]   {r.patent_id}  score={score}")

        return final_result

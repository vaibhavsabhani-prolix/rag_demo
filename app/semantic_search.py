import time
from collections.abc import Callable
from typing import TypeVar

from app.models.collection import SearchCollection
from app.models.parsed_query import ParsedQuery
from app.models.scoring import FinalSearchResult
from app.query_understanding.engine import QueryUnderstandingEngine
from app.reranking.reranker import BGEReranker
from app.retrieval.metadata_filter import filter_candidates
from app.retrieval.retriever import CandidateRetriever
from app.scoring.scorer import FinalScorer
from app.verification.verifier import RelationshipVerifier

T = TypeVar("T")

PhaseCallback = Callable[[int, str, object, float], None]

PHASE_NAMES = {
    1: "Query Understanding",
    2: "Candidate Vector Retrieval",
    3: "Metadata Filtering",
    4: "Semantic Relationship Verification",
    5: "BGE Cross-Encoder Reranking",
    6: "Final Patent Scoring",
}


class SearchPipeline:

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
        on_phase_complete: PhaseCallback | None,
        phase_num: int,
        result: object,
        elapsed_ms: float,
    ) -> None:
        if on_phase_complete is not None:
            on_phase_complete(phase_num, PHASE_NAMES[phase_num], result, elapsed_ms)

    def _run_phase(
        self,
        phase_num: int,
        operation: Callable[[], T],
        on_phase_complete: PhaseCallback | None,
    ) -> T:
        start = time.perf_counter()

        result = operation()

        elapsed_ms = (time.perf_counter() - start) * 1000

        self._emit(
            on_phase_complete,
            phase_num,
            result,
            elapsed_ms,
        )

        return result

    def run(
        self,
        query: str,
        collection: SearchCollection,
        use_cache: bool = True,
        on_phase_complete: PhaseCallback | None = None,
    ) -> FinalSearchResult:

        parsed_query = self._run_phase(
            1,
            lambda: self.engine.parse(query, use_cache=use_cache),
            on_phase_complete,
        )

        return self.run_parsed(
            parsed_query,
            collection,
            on_phase_complete,
        )

    def run_parsed(
        self,
        parsed_query: ParsedQuery,
        collection: SearchCollection,
        on_phase_complete: PhaseCallback | None = None,
    ) -> FinalSearchResult:

        # Phase 2 — Candidate Retrieval (Vector Search)
        retrieval_result = self._run_phase(
            2,
            lambda: self.retriever.retrieve_candidates(
                parsed_query,
                collection,
            ),
            on_phase_complete,
        )

        # Phase 3 — Metadata Filtering & Constraint Enforcement
        filtered_result = self._run_phase(
            3,
            lambda: filter_candidates(
                candidates=retrieval_result.candidates,
                metadata_filters=parsed_query.metadata_filters,
                is_metadata_only=parsed_query.is_metadata_only,
            ),
            on_phase_complete,
        )

        # Phase 4 — Semantic Relationship & Requirement Verification
        verification_result = self._run_phase(
            4,
            lambda: self.verifier.verify_candidates(
                parsed_query,
                filtered_result.candidates,
            ),
            on_phase_complete,
        )

        # Phase 5 — BGE Cross-Encoder Reranking
        rerank_result = self._run_phase(
            5,
            lambda: self.reranker.rerank_candidates(
                parsed_query,
                verification_result,
                filtered_result.candidates,
            ),
            on_phase_complete,
        )

        # Phase 6 — Final Patent Scoring & Result Selection
        final_result = self._run_phase(
            6,
            lambda: self.scorer.score_and_rank(
                rerank_result,
                is_metadata_only=parsed_query.is_metadata_only,
            ),
            on_phase_complete,
        )

        for r in final_result.results:
            score = "unscored" if r.final_score is None else f"{r.final_score:.2f}"
            print(f"[FinalResult]   {r.patent_id}  score={score}")

        return final_result

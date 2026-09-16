"""
Phase 7: Final Patent Scoring & Result Selection Engine

Computes deterministic, multi-signal final patent scores (0.0 to 10.0 scale)
combining relationship verification, requirement satisfaction, BGE cross-encoder
reranking, and retrieval evidence quality. Enforces FINAL_SCORE_THRESHOLD filtering
and produces ranked, bounded final results with full explainability.

Key guarantees:
- Pure in-memory deterministic computation (zero LLM calls, zero BGE calls, zero embeddings, zero Qdrant queries).
- Score range strictly normalized to 0.0 <= final_score <= 10.0 (rounded to 2 decimal places).
- Configurable threshold enforcement: final_score >= FINAL_SCORE_THRESHOLD included; strictly excluded otherwise.
- Strict weight validation: weights must sum to 1.0 (within 1e-6 tolerance).
- Descending score sorting; all qualifying patents returned (no fixed count cap).
- Complete explainability via ScoreBreakdown.
- Full preservation of Phase 5 verification and Phase 6 evidence data.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import (
    FINAL_SCORE_THRESHOLD,
    FINAL_WEIGHT_RELATIONSHIP,
    FINAL_WEIGHT_REQUIREMENT,
    FINAL_WEIGHT_RERANKER,
    FINAL_WEIGHT_RETRIEVAL,
)
from app.models.reranking import RerankBatchResult, RerankedPatentResult
from app.models.scoring import (
    FinalPatentResult,
    FinalSearchResult,
    ScoreBreakdown,
)

logger = logging.getLogger(__name__)


def validate_weights(
    w_rel: float,
    w_req: float,
    w_rerank: float,
    w_ret: float,
) -> None:
    """
    Validate that multi-signal scoring weights sum to 1.0 within a small floating-point tolerance.
    """
    total = w_rel + w_req + w_rerank + w_ret
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Scoring weights must sum to 1.0 (got {total:.6f}: "
            f"relationship={w_rel}, requirement={w_req}, reranker={w_rerank}, retrieval={w_ret})"
        )


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value between min_val and max_val."""
    return max(min_val, min(max_val, float(value)))


class FinalScorer:
    """
    Phase 7 Final Patent Scoring and Result Selection Engine.
    """

    def __init__(
        self,
        score_threshold: float = FINAL_SCORE_THRESHOLD,
        weight_relationship: float = FINAL_WEIGHT_RELATIONSHIP,
        weight_requirement: float = FINAL_WEIGHT_REQUIREMENT,
        weight_reranker: float = FINAL_WEIGHT_RERANKER,
        weight_retrieval: float = FINAL_WEIGHT_RETRIEVAL,
    ):
        self.score_threshold = float(score_threshold)
        self.weight_relationship = float(weight_relationship)
        self.weight_requirement = float(weight_requirement)
        self.weight_reranker = float(weight_reranker)
        self.weight_retrieval = float(weight_retrieval)

        # Validate weights at initialization
        validate_weights(
            self.weight_relationship,
            self.weight_requirement,
            self.weight_reranker,
            self.weight_retrieval,
        )

    def calculate_patent_score(
        self,
        patent: RerankedPatentResult,
    ) -> Tuple[float, ScoreBreakdown]:
        """
        Calculate normalized component scores and the final patent score on a 0.0 to 10.0 scale.
        """
        # 1. Relationship component (0.0 to 1.0)
        # If any explicit contradiction exists, penalize relationship coverage
        rel_coverage = clamp(patent.relationship_coverage, 0.0, 1.0)
        if patent.contradicted_count > 0:
            # Explicit contradiction strongly penalizes relationship score
            rel_score = 0.0
        else:
            rel_score = rel_coverage

        # 2. Requirement component (0.0 to 1.0)
        req_score = clamp(patent.requirement_coverage, 0.0, 1.0)

        # 3. Reranker component (0.0 to 1.0)
        rerank_score = clamp(patent.best_reranker_score, 0.0, 1.0)

        # 4. Retrieval component (0.0 to 1.0)
        ret_score = clamp(patent.candidate_score, 0.0, 1.0)

        # Calculate weighted contributions (each on 0.0 to 10.0 scale for clarity)
        rel_contrib = self.weight_relationship * rel_score * 10.0
        req_contrib = self.weight_requirement * req_score * 10.0
        rerank_contrib = self.weight_reranker * rerank_score * 10.0
        ret_contrib = self.weight_retrieval * ret_score * 10.0

        raw_total = rel_contrib + req_contrib + rerank_contrib + ret_contrib
        final_score = round(clamp(raw_total, 0.0, 10.0), 2)

        breakdown = ScoreBreakdown(
            relationship_score=round(rel_score, 4),
            requirement_score=round(req_score, 4),
            reranker_score=round(rerank_score, 4),
            retrieval_score=round(ret_score, 4),
            relationship_weighted=round(rel_contrib, 3),
            requirement_weighted=round(req_contrib, 3),
            reranker_weighted=round(rerank_contrib, 3),
            retrieval_weighted=round(ret_contrib, 3),
            raw_total=round(raw_total, 3),
        )

        return final_score, breakdown

    def score_and_rank(
        self,
        rerank_batch: RerankBatchResult,
    ) -> FinalSearchResult:
        """
        Evaluate all candidate patents from Phase 6, compute final scores,
        filter by FINAL_SCORE_THRESHOLD, and sort descending. All qualifying
        patents are returned (no fixed result count cap).
        """
        t_start = time.perf_counter()

        candidates = rerank_batch.reranked_patents
        if not candidates:
            total_time_ms = (time.perf_counter() - t_start) * 1000
            return FinalSearchResult(
                results=[],
                total_candidates_evaluated=0,
                passed_threshold_count=0,
                rejected_count=0,
                threshold_used=self.score_threshold,
                weights_used={
                    "relationship": self.weight_relationship,
                    "requirement": self.weight_requirement,
                    "reranker": self.weight_reranker,
                    "retrieval": self.weight_retrieval,
                },
                timings={"scoring_ms": 0.0, "filter_sort_ms": 0.0, "total_ms": round(total_time_ms, 3)},
            )

        # 1. Compute scores for all candidates
        t_scoring_start = time.perf_counter()
        scored_patents: List[FinalPatentResult] = []

        for rpat in candidates:
            final_score, breakdown = self.calculate_patent_score(rpat)
            scored_patents.append(
                FinalPatentResult(
                    patent_id=rpat.patent_id,
                    final_score=final_score,
                    score_breakdown=breakdown,
                    metadata=rpat.metadata,
                    relationship_coverage=rpat.relationship_coverage,
                    requirement_coverage=rpat.requirement_coverage,
                    best_reranker_score=rpat.best_reranker_score,
                    candidate_score=rpat.candidate_score,
                    supported_count=rpat.supported_count,
                    unsupported_count=rpat.unsupported_count,
                    contradicted_count=rpat.contradicted_count,
                    unknown_count=rpat.unknown_count,
                    relationships=rpat.relationships,
                    requirements=rpat.requirements,
                    evidence=rpat.evidence,
                )
            )

        scoring_time_ms = (time.perf_counter() - t_scoring_start) * 1000

        # 2. Filter by threshold (score >= FINAL_SCORE_THRESHOLD) and sort descending
        t_filter_start = time.perf_counter()
        qualifying = [p for p in scored_patents if p.final_score >= self.score_threshold]
        qualifying.sort(key=lambda p: p.final_score, reverse=True)
        top_results = qualifying

        filter_sort_time_ms = (time.perf_counter() - t_filter_start) * 1000
        total_time_ms = (time.perf_counter() - t_start) * 1000

        passed_count = len(qualifying)
        rejected_count = len(scored_patents) - passed_count

        return FinalSearchResult(
            results=top_results,
            total_candidates_evaluated=len(scored_patents),
            passed_threshold_count=passed_count,
            rejected_count=rejected_count,
            threshold_used=self.score_threshold,
            weights_used={
                "relationship": self.weight_relationship,
                "requirement": self.weight_requirement,
                "reranker": self.weight_reranker,
                "retrieval": self.weight_retrieval,
            },
            timings={
                "scoring_ms": round(scoring_time_ms, 3),
                "filter_sort_ms": round(filter_sort_time_ms, 3),
                "total_ms": round(total_time_ms, 3),
            },
        )

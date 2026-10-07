"""
Phase 4: High-Speed Hypothesis-Level Relationship & Requirement Evidence Scoring

Performs fast (< 500ms), GPU-accelerated semantic scoring of candidate patent
evidence against requested directed relationships and requirements using
cross-encoder semantic entailment.

Every relationship and every requirement is an independent hypothesis, scored
against every chunk of every surviving Phase-3 candidate. Phase 4 does NOT
eliminate candidates — it annotates each one with per-hypothesis support and
patent-level coverage ratios, and every candidate it evaluates proceeds to
Phase 5/6, which combine that coverage with reranker/retrieval signals to
decide final ranking.

The hypothesis count is not fixed (it's relationships + requirements, however
many the query produced), so all (hypothesis, batch) scoring work is flattened
into a single global queue consumed by one bounded worker pool capped at
RERANK_CONCURRENT_REQUESTS total in flight — never `hypothesis_count *
concurrent_requests`. Batches from different hypotheses interleave (round-
robin by batch position) rather than running one hypothesis to completion
before the next starts, so wall-clock time stops scaling linearly with
hypothesis count.

Eliminates slow LLM latency while maintaining high precision for compositional
patent matching.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from app.config import (
    RERANK_BATCH_SIZE,
    RERANK_CONCURRENT_REQUESTS,
    RERANKER_REMOTE_API_KEY,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RERANKER_REQUEST_TIMEOUT,
    VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD,
    VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD,
)
from app.models.candidate import CandidateChunk, CandidatePatent
from app.models.parsed_query import ParsedQuery, SemanticRelationship
from app.models.verification import (
    PatentVerificationResult,
    RelationshipVerification,
    RequirementVerification,
    VerificationBatchResult,
)

logger = logging.getLogger(__name__)


def calculate_relationship_coverage(verifications: List[RelationshipVerification]) -> float:
    """Compute local deterministic relationship coverage ratio (0.0 to 1.0)."""
    if not verifications:
        return 1.0
    supported = sum(1 for v in verifications if v.supported)
    return round(supported / len(verifications), 4)


def calculate_requirement_coverage(verifications: List[RequirementVerification]) -> float:
    """Compute local deterministic requirement coverage ratio (0.0 to 1.0)."""
    if not verifications:
        return 1.0
    supported = sum(1 for v in verifications if v.supported)
    return round(supported / len(verifications), 4)


def _tokenize_terms(text: str) -> List[str]:
    """Extract lowercase alphanumeric words from text."""
    return [w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]+\b", text) if len(w) > 2]


class RelationshipVerifier:
    """
    Phase 4 Fast Hypothesis-Level Relationship & Requirement Evidence Scorer.
    Uses GPU Cross-Encoder Entailment. Scores, never eliminates.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        batch_size: int = RERANK_BATCH_SIZE,
        concurrent_requests: int = RERANK_CONCURRENT_REQUESTS,
    ):
        self.base_url = (base_url or RERANKER_REMOTE_BASE_URL).rstrip("/")
        self.model = model or RERANKER_REMOTE_MODEL
        self.api_key = api_key or RERANKER_REMOTE_API_KEY
        self.timeout = timeout if timeout is not None else RERANKER_REQUEST_TIMEOUT
        self.batch_size = batch_size
        self.concurrent_requests = concurrent_requests

        # Persistent requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _score_batch(self, query: str, documents: List[str]) -> List[float]:
        """Send a single batch of (query, doc) pairs to the remote BGE cross-encoder."""
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }

        endpoints = [f"{self.base_url}/v1/rerank", f"{self.base_url}/rerank"]
        for endpoint in endpoints:
            try:
                resp = self.session.post(endpoint, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    sorted_results = sorted(results, key=lambda x: x.get("index", 0))
                    scores = [float(item.get("relevance_score", 0.0)) for item in sorted_results]
                    if len(scores) == len(documents):
                        return scores
                    score_map = {item.get("index"): float(item.get("relevance_score", 0.0)) for item in results}
                    return [score_map.get(i, 0.0) for i in range(len(documents))]
            except Exception as e:
                logger.warning("Fast verification cross-encoder exception at %s: %s", endpoint, e)

        return [0.0] * len(documents)

    @staticmethod
    def _build_score_tasks(
        hypotheses: List[Tuple[str, int, str]],
        starts: List[int],
    ) -> List[Tuple[str, int, str, int]]:
        """
        Flatten every (hypothesis, batch) pair into one globally-ordered task
        list: *starts* is the outer loop and *hypotheses* the inner loop, so
        task order is round-robin across hypotheses (one batch from every
        hypothesis, then the next batch from every hypothesis, ...) rather
        than grouped hypothesis-by-hypothesis. Submitted to a single bounded
        worker pool, this round-robin order is what lets batches from
        different hypotheses execute concurrently instead of one hypothesis
        having to finish before the next one starts.
        """
        return [
            (kind, idx, hyp_text, start)
            for start in starts
            for kind, idx, hyp_text in hypotheses
        ]

    def _score_all_hypotheses(
        self,
        hypotheses: List[Tuple[str, int, str]],
        doc_texts: List[str],
        all_chunks: List[Tuple[str, CandidateChunk]],
        rel_scores: Dict[Tuple[int, str, int], float],
        req_scores: Dict[Tuple[int, str, int], float],
    ) -> None:
        """
        Score every hypothesis against every chunk through a single global
        work queue and one bounded worker pool (max self.concurrent_requests
        BGE HTTP requests in flight at once, regardless of how many
        hypotheses there are - never hypothesis_count * concurrent_requests).
        Preserves the existing batch size; only the scheduling changes.
        """
        if not doc_texts or not hypotheses:
            return

        starts = list(range(0, len(doc_texts), self.batch_size))
        tasks = self._build_score_tasks(hypotheses, starts)
        lock = threading.Lock()

        def run(kind: str, idx: int, hyp_text: str, start: int) -> None:
            batch_docs = doc_texts[start : start + self.batch_size]
            batch_chunks = all_chunks[start : start + self.batch_size]
            batch_scores = self._score_batch(hyp_text, batch_docs)
            target = rel_scores if kind == "relationship" else req_scores
            with lock:
                for (pid, ch), sc in zip(batch_chunks, batch_scores):
                    target[(idx, pid, ch.chunk_id)] = sc

        if len(tasks) == 1:
            run(*tasks[0])
            return

        max_workers = min(self.concurrent_requests, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run, *task) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error("Error scoring Phase 4 batch: %s", e)

    def verify_candidates(
        self,
        parsed_query: ParsedQuery,
        candidates: List[CandidatePatent],
    ) -> VerificationBatchResult:

        t_start = time.perf_counter()

        target_candidates = candidates

        if not target_candidates:
            total_time_ms = (time.perf_counter() - t_start) * 1000
            return VerificationBatchResult(
                verified_patents=[],
                total_evaluated=0,
                fully_supported_count=0,
                partially_supported_count=0,
                unsupported_count=0,
                timings={"total_ms": round(total_time_ms, 2)},
            )

        relationships = parsed_query.relationships or []
        requirements = parsed_query.requirements or []

        # If query is metadata-only or has no relationships & no requirements
        if parsed_query.is_metadata_only or (not relationships and not requirements):
            verified: List[PatentVerificationResult] = [
                PatentVerificationResult(
                    patent_id=c.patent_id,
                    relationships=[],
                    requirements=[],
                    relationship_coverage=1.0,
                    requirement_coverage=1.0,
                    supported_count=0,
                    unsupported_count=0,
                    contradicted_count=0,
                    unknown_count=0,
                    metadata=c.metadata,
                    candidate_score=c.retrieval_score,
                )
                for c in target_candidates
            ]
            total_time_ms = (time.perf_counter() - t_start) * 1000
            return VerificationBatchResult(
                verified_patents=verified,
                total_evaluated=len(verified),
                fully_supported_count=len(verified),
                partially_supported_count=0,
                unsupported_count=0,
                timings={"total_ms": round(total_time_ms, 2)},
            )

        # 1. Prepare verification hypotheses - however many relationships and
        # requirements this query happens to have produced (not a fixed count).
        # rel_hypotheses: list of (idx, hyp_text, subject, object, relation)
        rel_hypotheses: List[Tuple[int, str, str, str, str]] = []
        for idx, r in enumerate(relationships):
            ctx_str = f" in {r.context}" if r.context else ""
            hyp = f"{r.subject} {r.relation} {r.object}{ctx_str}".strip()
            rel_hypotheses.append((idx, hyp, r.subject, r.object, r.relation))

        # req_hypotheses: list of (idx, req_text)
        req_hypotheses: List[Tuple[int, str]] = [(idx, req.strip()) for idx, req in enumerate(requirements)]

        # all_hypotheses: unified (kind, idx, hyp_text) list consumed by the
        # global scheduler below - relationships and requirements compete for
        # the same bounded worker pool rather than running one kind, then the
        # other, sequentially.
        all_hypotheses: List[Tuple[str, int, str]] = [
            ("relationship", idx, hyp_text) for idx, hyp_text, _, _, _ in rel_hypotheses
        ] + [("requirement", idx, req_text) for idx, req_text in req_hypotheses]

        # 2. Gather all candidate evidence chunks
        # Map: (patent_id, chunk_id) -> CandidateChunk
        all_chunks: List[Tuple[str, CandidateChunk]] = []
        for cand in target_candidates:
            for ch in cand.chunks:
                all_chunks.append((cand.patent_id, ch))

        doc_texts = [f"Section: {ch.section}\n\n{ch.text or ''}" if ch.section else (ch.text or "") for _, ch in all_chunks]

        # 3. Score every hypothesis against every chunk through one global,
        # boundedly-concurrent BGE work queue (see _score_all_hypotheses).
        # rel_scores / req_scores: (hyp_idx, patent_id, chunk_id) -> score
        rel_scores: Dict[Tuple[int, str, int], float] = {}
        req_scores: Dict[Tuple[int, str, int], float] = {}
        self._score_all_hypotheses(all_hypotheses, doc_texts, all_chunks, rel_scores, req_scores)

        # 5. Evaluate each Candidate Patent
        verified_patents: List[PatentVerificationResult] = []
        eliminated_patents: List[PatentVerificationResult] = []
        eliminated_count = 0

        for cand in target_candidates:
            pid = cand.patent_id
            chunks = cand.chunks

            # --- Evaluate Relationships ---
            rel_verifications: List[RelationshipVerification] = []
            for r_idx, hyp_text, subj, obj, rel in rel_hypotheses:
                best_score = 0.0
                best_cid: Optional[int] = None

                for ch in chunks:
                    sc = rel_scores.get((r_idx, pid, ch.chunk_id), 0.0)
                    if sc > best_score:
                        best_score = sc
                        best_cid = ch.chunk_id

                # Relationship is supported purely by cross-encoder entailment score
                is_supported = best_score >= VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD
                if is_supported and best_cid is not None:
                    status = "SUPPORTED"
                    cids = [best_cid]
                    explanation = f"Verified in Chunk #{best_cid}"
                else:
                    status = "NOT_SUPPORTED"
                    cids = []
                    explanation = "No evidence chunk sufficiently connects subject and object."

                rel_verifications.append(
                    RelationshipVerification(
                        relationship_index=r_idx,
                        subject=subj,
                        relation=rel,
                        object=obj,
                        supported=is_supported,
                        status=status,
                        score=round(best_score, 4),
                        evidence_chunk_ids=cids,
                        explanation=explanation,
                    )
                )

            # --- Evaluate Requirements ---
            req_verifications: List[RequirementVerification] = []
            for req_idx, req_text in req_hypotheses:
                best_req_score = 0.0
                best_req_cid: Optional[int] = None

                req_words = set(_tokenize_terms(req_text))

                for ch in chunks:
                    sc = req_scores.get((req_idx, pid, ch.chunk_id), 0.0)
                    ch_words = set(_tokenize_terms(ch.text or ""))
                    overlap = len(req_words & ch_words) / len(req_words) if req_words else 0.0

                    effective_score = max(sc, overlap * 0.5)
                    if effective_score > best_req_score:
                        best_req_score = effective_score
                        best_req_cid = ch.chunk_id

                is_req_supported = best_req_score >= VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD
                if is_req_supported and best_req_cid is not None:
                    req_verifications.append(
                        RequirementVerification(
                            requirement_index=req_idx,
                            requirement=req_text,
                            supported=True,
                            score=round(best_req_score, 4),
                            evidence_chunk_ids=[best_req_cid],
                            explanation=f"Requirement supported in Chunk #{best_req_cid}",
                        )
                    )
                else:
                    req_verifications.append(
                        RequirementVerification(
                            requirement_index=req_idx,
                            requirement=req_text,
                            supported=False,
                            score=round(best_req_score, 4),
                            evidence_chunk_ids=[],
                            explanation="Requirement not satisfied in candidate evidence.",
                        )
                    )

            # Per-patent coverage ratios - reported for display and consumed by
            # Phase 6's weighted score. Phase 4 no longer eliminates candidates
            # on these: every candidate handed to it proceeds to Phase 5/6, which
            # weigh coverage alongside the reranker/retrieval signals instead.
            rel_cov = calculate_relationship_coverage(rel_verifications)
            req_cov = calculate_requirement_coverage(req_verifications)

            sup_count = sum(1 for r in rel_verifications if r.supported)
            unsup_count = sum(1 for r in rel_verifications if r.status == "NOT_SUPPORTED")
            contra_count = sum(1 for r in rel_verifications if r.status == "CONTRADICTED")
            unk_count = sum(1 for r in rel_verifications if r.status == "UNKNOWN")

            result = PatentVerificationResult(
                patent_id=pid,
                qualified=True,
                relationships=rel_verifications,
                requirements=req_verifications,
                relationship_coverage=rel_cov,
                requirement_coverage=req_cov,
                supported_count=sup_count,
                unsupported_count=unsup_count,
                contradicted_count=contra_count,
                unknown_count=unk_count,
                metadata=cand.metadata,
                candidate_score=cand.retrieval_score,
            )

            verified_patents.append(result)

        total_time_ms = (time.perf_counter() - t_start) * 1000
        avg_ms = total_time_ms / len(target_candidates) if target_candidates else 0.0

        fully_sup = sum(1 for r in verified_patents if r.relationship_coverage >= 1.0)
        partially_sup = sum(1 for r in verified_patents if 0.0 < r.relationship_coverage < 1.0)
        unsup = sum(1 for r in verified_patents if r.relationship_coverage == 0.0)

        return VerificationBatchResult(
            verified_patents=verified_patents,
            eliminated_patents=eliminated_patents,
            total_evaluated=len(verified_patents),
            fully_supported_count=fully_sup,
            partially_supported_count=partially_sup,
            unsupported_count=unsup,
            eliminated_count=eliminated_count,
            timings={
                "verification_ms": round(total_time_ms, 2),
                "total_ms": round(total_time_ms, 2),
                "avg_per_patent_ms": round(avg_ms, 2),
            },
        )

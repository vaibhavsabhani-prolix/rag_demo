"""
Phase 5: High-Speed Semantic Relationship & Requirement Verification Engine

Performs fast (< 500ms), GPU-accelerated semantic verification of candidate
patent evidence against requested directed relationships and requirements using
cross-encoder semantic entailment and deterministic span proximity.

Eliminates slow LLM latency while maintaining high precision for compositional
patent matching.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
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
    VERIFICATION_MAX_CANDIDATES,
    VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD,
    VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD,
)
from app.models.evidence import EvidenceChunk, EvidenceRetrievalResult, PatentEvidence
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


def check_span_proximity(subject: str, object_: str, text: str, max_word_distance: int = 40) -> Tuple[bool, Optional[str]]:
    """
    Check if subject terms and object terms co-occur within a bounded word distance in text.
    Returns (is_cooccurring, matching_snippet).
    """
    if not subject or not object_ or not text:
        return False, None

    subj_words = set(_tokenize_terms(subject))
    obj_words = set(_tokenize_terms(object_))

    if not subj_words or not obj_words:
        return False, None

    words = text.split()
    text_lower = text.lower()

    # Exact phrase substring check
    if subject.lower() in text_lower and object_.lower() in text_lower:
        # Find sentence containing both
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_low = s.lower()
            if any(sw in s_low for sw in subj_words) and any(ow in s_low for ow in obj_words):
                return True, s.strip()[:150]

    # Sliding window proximity check
    subj_indices = [i for i, w in enumerate(words) if any(sw in w.lower() for sw in subj_words)]
    obj_indices = [i for i, w in enumerate(words) if any(ow in w.lower() for ow in obj_words)]

    if not subj_indices or not obj_indices:
        return False, None

    min_dist = min(abs(si - oi) for si in subj_indices for oi in obj_indices)
    if min_dist <= max_word_distance:
        # Extract snippet around match
        best_si = min(subj_indices, key=lambda si: min(abs(si - oi) for oi in obj_indices))
        start = max(0, best_si - 10)
        end = min(len(words), best_si + max_word_distance + 10)
        snippet = " ".join(words[start:end])
        return True, snippet[:150]

    return False, None


class RelationshipVerifier:
    """
    Phase 5 Fast Semantic Relationship Verification Engine.
    Uses GPU Cross-Encoder Entailment + Deterministic Span Proximity.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_candidates: int = VERIFICATION_MAX_CANDIDATES,
        batch_size: int = RERANK_BATCH_SIZE,
        concurrent_requests: int = RERANK_CONCURRENT_REQUESTS,
    ):
        self.base_url = (base_url or RERANKER_REMOTE_BASE_URL).rstrip("/")
        self.model = model or RERANKER_REMOTE_MODEL
        self.api_key = api_key or RERANKER_REMOTE_API_KEY
        self.timeout = timeout if timeout is not None else RERANKER_REQUEST_TIMEOUT
        self.max_candidates = max_candidates
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

    def _score_pairs_remote(self, query: str, documents: List[str]) -> List[float]:
        """Send a batch of (query, doc) pairs to the remote BGE cross-encoder."""
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

    def verify_candidates(
        self,
        parsed_query: ParsedQuery,
        evidence_result: EvidenceRetrievalResult,
        max_candidates: Optional[int] = None,
    ) -> VerificationBatchResult:
        """
        Execute high-speed relationship and requirement verification across all candidate patents.
        Runs in < 300ms using GPU cross-encoder entailment and span proximity.
        """
        t_start = time.perf_counter()

        limit = max_candidates or self.max_candidates
        target_candidates = evidence_result.patent_evidence_list[:limit]

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
                    candidate_score=c.candidate_score,
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

        # 1. Prepare verification hypotheses
        # rel_hypotheses: list of (idx, hyp_text, subject, object, relation)
        rel_hypotheses: List[Tuple[int, str, str, str, str]] = []
        for idx, r in enumerate(relationships):
            ctx_str = f" in {r.context}" if r.context else ""
            hyp = f"{r.subject} {r.relation} {r.object}{ctx_str}".strip()
            rel_hypotheses.append((idx, hyp, r.subject, r.object, r.relation))

        # req_hypotheses: list of (idx, req_text)
        req_hypotheses: List[Tuple[int, str]] = [(idx, req.strip()) for idx, req in enumerate(requirements)]

        # 2. Gather all candidate evidence chunks
        # Map: (patent_id, chunk_id) -> EvidenceChunk
        all_chunks: List[Tuple[str, EvidenceChunk]] = []
        for cand in target_candidates:
            for ch in cand.chunks:
                all_chunks.append((cand.patent_id, ch))

        # 3. Fast Batched Scoring for Relationships via Cross-Encoder
        # Score each relationship hypothesis against all chunks
        # rel_scores: (rel_idx, patent_id, chunk_id) -> score
        rel_scores: Dict[Tuple[int, str, int], float] = {}
        doc_texts = [f"Section: {ch.section}\n\n{ch.text[:1000]}" if ch.section else ch.text[:1000] for _, ch in all_chunks]

        if rel_hypotheses and doc_texts:
            for r_idx, hyp_text, _, _, _ in rel_hypotheses:
                scores = self._score_pairs_remote(hyp_text, doc_texts)
                for (pid, ch), sc in zip(all_chunks, scores):
                    rel_scores[(r_idx, pid, ch.chunk_id)] = sc

        # 4. Fast Batched Scoring for Requirements via Cross-Encoder
        req_scores: Dict[Tuple[int, str, int], float] = {}
        if req_hypotheses and doc_texts:
            for req_idx, req_text in req_hypotheses:
                scores = self._score_pairs_remote(req_text, doc_texts)
                for (pid, ch), sc in zip(all_chunks, scores):
                    req_scores[(req_idx, pid, ch.chunk_id)] = sc

        # 5. Evaluate each Candidate Patent
        verified_patents: List[PatentVerificationResult] = []

        for cand in target_candidates:
            pid = cand.patent_id
            chunks = cand.chunks

            # --- Evaluate Relationships ---
            rel_verifications: List[RelationshipVerification] = []
            for r_idx, hyp_text, subj, obj, rel in rel_hypotheses:
                best_score = 0.0
                best_cid: Optional[int] = None
                best_snippet: Optional[str] = None
                has_proximity = False

                for ch in chunks:
                    sc = rel_scores.get((r_idx, pid, ch.chunk_id), 0.0)
                    if sc > best_score:
                        best_score = sc
                        best_cid = ch.chunk_id

                    # Check span proximity
                    cooccurs, snippet = check_span_proximity(subj, obj, ch.text)
                    if cooccurs:
                        has_proximity = True
                        if best_snippet is None or sc >= best_score:
                            best_snippet = snippet
                            best_cid = ch.chunk_id

                # Relationship is supported if proximity matches or cross-encoder score indicates positive entailment
                is_supported = has_proximity or best_score >= VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD
                if is_supported and best_cid is not None:
                    status = "SUPPORTED"
                    confidence = round(max(0.75, min(0.98, best_score * 2.0 if best_score > 0 else 0.85)), 2)
                    cids = [best_cid]
                    explanation = f"Verified in Chunk #{best_cid}" + (f": \"{best_snippet}\"" if best_snippet else "")
                else:
                    status = "NOT_SUPPORTED"
                    confidence = 0.80
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
                        confidence=confidence,
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
                    ch_words = set(_tokenize_terms(ch.text))
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
                            confidence=round(max(0.75, min(0.95, best_req_score * 2.0)), 2),
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
                            confidence=0.80,
                            evidence_chunk_ids=[],
                            explanation="Requirement not satisfied in candidate evidence.",
                        )
                    )

            # Local Coverage Metrics
            rel_cov = calculate_relationship_coverage(rel_verifications)
            req_cov = calculate_requirement_coverage(req_verifications)

            sup_count = sum(1 for r in rel_verifications if r.supported)
            unsup_count = sum(1 for r in rel_verifications if r.status == "NOT_SUPPORTED")
            contra_count = sum(1 for r in rel_verifications if r.status == "CONTRADICTED")
            unk_count = sum(1 for r in rel_verifications if r.status == "UNKNOWN")

            verified_patents.append(
                PatentVerificationResult(
                    patent_id=pid,
                    relationships=rel_verifications,
                    requirements=req_verifications,
                    relationship_coverage=rel_cov,
                    requirement_coverage=req_cov,
                    supported_count=sup_count,
                    unsupported_count=unsup_count,
                    contradicted_count=contra_count,
                    unknown_count=unk_count,
                    metadata=cand.metadata,
                    candidate_score=cand.candidate_score,
                )
            )

        total_time_ms = (time.perf_counter() - t_start) * 1000
        avg_ms = total_time_ms / len(target_candidates) if target_candidates else 0.0

        fully_sup = sum(1 for r in verified_patents if r.relationship_coverage >= 1.0)
        partially_sup = sum(1 for r in verified_patents if 0.0 < r.relationship_coverage < 1.0)
        unsup = sum(1 for r in verified_patents if r.relationship_coverage == 0.0)

        return VerificationBatchResult(
            verified_patents=verified_patents,
            total_evaluated=len(verified_patents),
            fully_supported_count=fully_sup,
            partially_supported_count=partially_sup,
            unsupported_count=unsup,
            timings={
                "verification_ms": round(total_time_ms, 2),
                "total_ms": round(total_time_ms, 2),
                "avg_per_patent_ms": round(avg_ms, 2),
            },
        )

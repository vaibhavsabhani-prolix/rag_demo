"""
Phase 6: BGE Cross-Encoder Reranking Engine

Evaluates and scores candidate patent evidence chunks against the user's complete
semantic intent using the BGE cross-encoder reranker (BAAI/bge-reranker-v2-m3).

Key guarantees:
- Consumes bounded candidates from Phase 5 and bounded evidence from Phase 4.
- Deterministic query construction (semantic query + relationships + requirements).
- Zero LLM calls and zero new embedding generations.
- Strict 4096-token combined limit enforcement (query + doc + prefix + safety margin <= 4096).
- Batch processing with bounded concurrency (requests.Session + ThreadPoolExecutor).
- Full preservation of Phase 5 relationship & requirement verification outcomes.
- Chunk-level scoring and patent-level intermediate aggregation (best_reranker_score).
- Zero final ranking weights calculated (reserved for Phase 7).
- Resilient error handling (failures do not crash search).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.chunking.token_counter import TokenCounter
from app.config import (
    RERANK_BATCH_SIZE,
    RERANK_CONCURRENT_REQUESTS,
    RERANKER_MAX_CONTEXT_TOKENS,
    RERANKER_REMOTE_API_KEY,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RERANKER_REQUEST_TIMEOUT,
    RERANKER_TOKEN_SAFETY_MARGIN,
)
from app.models.evidence import EvidenceChunk, EvidenceRetrievalResult
from app.models.parsed_query import ParsedQuery
from app.models.reranking import (
    RerankBatchResult,
    RerankedEvidenceChunk,
    RerankedPatentResult,
)
from app.models.verification import PatentVerificationResult, VerificationBatchResult

logger = logging.getLogger(__name__)


def build_rerank_query(parsed_query: ParsedQuery) -> str:
    """
    Construct a compact, deterministic, information-dense reranking query
    from ParsedQuery components without any LLM calls or keyword reduction.
    """
    parts: List[str] = []

    if parsed_query.semantic_query:
        parts.append(parsed_query.semantic_query.strip())
    elif parsed_query.original_query:
        parts.append(parsed_query.original_query.strip())

    if parsed_query.relationships:
        rel_strs = [
            f"{r.subject} {r.relation} {r.object}" + (f" ({r.context})" if r.context else "")
            for r in parsed_query.relationships
        ]
        parts.append("Relationships: " + "; ".join(rel_strs))

    if parsed_query.requirements:
        parts.append("Requirements: " + "; ".join(parsed_query.requirements))

    return " \n".join(parts).strip()


class BGEReranker:
    """
    Phase 6 BGE Cross-Encoder Reranking Engine.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        batch_size: int = RERANK_BATCH_SIZE,
        concurrent_requests: int = RERANK_CONCURRENT_REQUESTS,
        max_context_tokens: int = RERANKER_MAX_CONTEXT_TOKENS,
        safety_margin: int = RERANKER_TOKEN_SAFETY_MARGIN,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.base_url = (base_url or RERANKER_REMOTE_BASE_URL).rstrip("/")
        self.model = model or RERANKER_REMOTE_MODEL
        self.api_key = api_key or RERANKER_REMOTE_API_KEY
        self.timeout = timeout if timeout is not None else RERANKER_REQUEST_TIMEOUT
        self.batch_size = batch_size
        self.concurrent_requests = concurrent_requests
        self.max_context_tokens = max_context_tokens
        self.safety_margin = safety_margin

        # Persistent requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        # Lazy / provided TokenCounter initialized for the reranker model
        self._token_counter = token_counter

    @property
    def token_counter(self) -> TokenCounter:
        """Lazily initialize the TokenCounter for the reranker model."""
        if self._token_counter is None:
            self._token_counter = TokenCounter(model_name=self.model)
        return self._token_counter

    def truncate_document_for_budget(
        self,
        text: str,
        section: Optional[str],
        doc_budget: int,
    ) -> Tuple[str, bool]:
        """
        Prepend section prefix (if present) and safely truncate document text
        so that (prefix + document) stays strictly within doc_budget tokens.

        Returns (final_formatted_doc, was_truncated).
        """
        prefix = f"Section: {section}\n\n" if section else ""
        prefix_tokens = self.token_counter.count(prefix) if prefix else 0
        avail_for_text = doc_budget - prefix_tokens

        if avail_for_text <= 0:
            # Prefix alone consumes budget — drop prefix or take small text
            logger.warning("Section prefix exceeds document token budget (%d tokens)", prefix_tokens)
            return text[:200], True

        text_tokens = self.token_counter.count(text)
        if text_tokens <= avail_for_text:
            return f"{prefix}{text}" if prefix else text, False

        # Document exceeds budget — safely slice at token boundary using offsets
        _, offsets = self.token_counter.tokenize_with_offsets(text)
        if len(offsets) > avail_for_text:
            end_char = offsets[avail_for_text - 1][1]
            truncated_text = text[:end_char]
        else:
            truncated_text = text

        final_doc = f"{prefix}{truncated_text}" if prefix else truncated_text
        return final_doc, True

    def _score_batch(
        self,
        query: str,
        documents: List[str],
        retry_on_token_error: bool = True,
        batch_label: str = "?",
    ) -> List[float]:
        """
        Send a single batch of documents to the remote BGE reranker endpoint.
        Returns a list of relevance scores aligned with input documents.
        """
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }

        # Try /v1/rerank, fallback to /rerank
        endpoints = [f"{self.base_url}/v1/rerank", f"{self.base_url}/rerank"]
        last_exception: Optional[Exception] = None

        for endpoint in endpoints:
            print(f"[Reranker] Request {batch_label} -> {endpoint} ({len(documents)} docs)")
            try:
                resp = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                print(f"[Reranker] Request {batch_label} <- HTTP {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    # Sort results by 'index' to guarantee exact alignment
                    sorted_results = sorted(results, key=lambda x: x.get("index", 0))
                    scores = [float(item.get("relevance_score", 0.0)) for item in sorted_results]
                    if len(scores) == len(documents):
                        return scores
                    else:
                        logger.warning(
                            "Reranker returned %d scores for %d documents",
                            len(scores),
                            len(documents),
                        )
                        # Build index map
                        score_map = {item.get("index"): float(item.get("relevance_score", 0.0)) for item in results}
                        return [score_map.get(i, 0.0) for i in range(len(documents))]

                elif resp.status_code == 400:
                    err_msg = resp.text
                    logger.warning("Reranker returned 400: %s", err_msg)
                    if retry_on_token_error and ("token" in err_msg.lower() or "context" in err_msg.lower()):
                        # Extra emergency truncation: reduce each document by 25% and retry once
                        logger.info("Retrying batch with emergency truncation...")
                        truncated_docs = []
                        for doc in documents:
                            _, offsets = self.token_counter.tokenize_with_offsets(doc)
                            keep_len = int(len(offsets) * 0.75)
                            if keep_len > 0 and len(offsets) > keep_len:
                                truncated_docs.append(doc[: offsets[keep_len - 1][1]])
                            else:
                                truncated_docs.append(doc[: len(doc) // 2])
                        return self._score_batch(query, truncated_docs, retry_on_token_error=False)

                    # Non-token 400 error
                    break

                else:
                    logger.warning("Reranker returned HTTP %d at %s", resp.status_code, endpoint)

            except Exception as e:
                last_exception = e
                logger.warning("Exception calling reranker at %s: %s", endpoint, e)

        logger.error(
            "All reranker endpoints failed for batch of %d documents. Error: %s",
            len(documents),
            last_exception,
        )
        # Resilient fallback: return 0.0 scores so pipeline continues
        return [0.0] * len(documents)

    def rerank_candidates(
        self,
        parsed_query: ParsedQuery,
        verification_result: VerificationBatchResult,
        evidence_result: EvidenceRetrievalResult,
    ) -> RerankBatchResult:
        """
        Execute Phase 6 BGE Reranking across all verified candidates and their bounded evidence chunks.
        """
        t_start = time.perf_counter()

        verified_patents = verification_result.verified_patents
        if not verified_patents:
            total_time_ms = (time.perf_counter() - t_start) * 1000
            return RerankBatchResult(
                reranked_patents=[],
                total_candidates=0,
                total_chunks_reranked=0,
                total_requests=0,
                truncated_chunks_count=0,
                reranking_query="",
                timings={"total_ms": round(total_time_ms, 2)},
            )

        # 1. Deterministically build reranking query
        reranking_query = build_rerank_query(parsed_query)
        query_tokens = self.token_counter.count(reranking_query)

        # 2. Compute safe document budget (combined query + doc + safety <= max_context_tokens)
        doc_budget = max(64, self.max_context_tokens - query_tokens - self.safety_margin)

        # 3. Gather evidence chunks per candidate patent
        evidence_by_patent: Dict[str, List[EvidenceChunk]] = evidence_result.evidence_by_patent

        # List of items to rerank: (patent_id, chunk_index_in_patent, EvidenceChunk, formatted_text, was_truncated)
        items_to_rerank: List[Tuple[str, int, EvidenceChunk, str, bool]] = []
        truncated_count = 0

        for vpat in verified_patents:
            pid = vpat.patent_id
            chunks = evidence_by_patent.get(pid, [])
            for c_idx, ch in enumerate(chunks):
                formatted_doc, was_trunc = self.truncate_document_for_budget(
                    text=ch.text or "",
                    section=ch.section,
                    doc_budget=doc_budget,
                )
                if was_trunc:
                    truncated_count += 1
                items_to_rerank.append((pid, c_idx, ch, formatted_doc, was_trunc))

        total_chunks = len(items_to_rerank)
        scores_map: Dict[Tuple[str, int], float] = {}
        total_requests = 0
        http_time_ms = 0.0

        # 4. Batch items and score concurrently
        if items_to_rerank:
            batches: List[List[Tuple[str, int, EvidenceChunk, str, bool]]] = []
            for i in range(0, total_chunks, self.batch_size):
                batches.append(items_to_rerank[i : i + self.batch_size])

            total_requests = len(batches)
            t_http_start = time.perf_counter()

            if len(batches) == 1:
                # Single batch optimization (no thread pool overhead)
                batch = batches[0]
                docs = [item[3] for item in batch]
                batch_scores = self._score_batch(reranking_query, docs, batch_label="1/1")
                for item, score in zip(batch, batch_scores):
                    pid, c_idx = item[0], item[1]
                    scores_map[(pid, c_idx)] = score
            else:
                with ThreadPoolExecutor(max_workers=min(self.concurrent_requests, len(batches))) as executor:
                    future_to_batch = {
                        executor.submit(
                            self._score_batch,
                            reranking_query,
                            [item[3] for item in batch],
                            batch_label=f"{i + 1}/{len(batches)}",
                        ): batch
                        for i, batch in enumerate(batches)
                    }
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            batch_scores = future.result()
                            for item, score in zip(batch, batch_scores):
                                pid, c_idx = item[0], item[1]
                                scores_map[(pid, c_idx)] = score
                        except Exception as e:
                            logger.error("Error scoring batch of chunks: %s", e)
                            for item in batch:
                                pid, c_idx = item[0], item[1]
                                scores_map[(pid, c_idx)] = 0.0

            http_time_ms = (time.perf_counter() - t_http_start) * 1000

        # 5. Assemble RerankedPatentResult preserving all Phase 5 verifications
        reranked_patents: List[RerankedPatentResult] = []

        for vpat in verified_patents:
            pid = vpat.patent_id
            chunks = evidence_by_patent.get(pid, [])
            reranked_chunks: List[RerankedEvidenceChunk] = []

            for c_idx, ch in enumerate(chunks):
                r_score = scores_map.get((pid, c_idx), 0.0)
                reranked_chunks.append(
                    RerankedEvidenceChunk(
                        patent_id=pid,
                        chunk_id=ch.chunk_id,
                        text=ch.text,
                        retrieval_score=ch.retrieval_score,
                        retrieval_source=ch.retrieval_source,
                        section=ch.section,
                        document_chunk_index=ch.document_chunk_index,
                        token_count=ch.token_count,
                        reranker_score=round(r_score, 6),
                    )
                )

            # Sort chunks by reranker_score descending
            reranked_chunks.sort(key=lambda c: c.reranker_score, reverse=True)

            best_r_score = max((c.reranker_score for c in reranked_chunks), default=0.0)
            avg_r_score = (
                sum(c.reranker_score for c in reranked_chunks) / len(reranked_chunks)
                if reranked_chunks
                else 0.0
            )

            reranked_patents.append(
                RerankedPatentResult(
                    patent_id=pid,
                    metadata=vpat.metadata,
                    candidate_score=vpat.candidate_score,
                    relationship_coverage=vpat.relationship_coverage,
                    requirement_coverage=vpat.requirement_coverage,
                    relationships=vpat.relationships,
                    requirements=vpat.requirements,
                    supported_count=vpat.supported_count,
                    unsupported_count=vpat.unsupported_count,
                    contradicted_count=vpat.contradicted_count,
                    unknown_count=vpat.unknown_count,
                    evidence=reranked_chunks,
                    best_reranker_score=round(best_r_score, 6),
                    avg_reranker_score=round(avg_r_score, 6),
                )
            )

        total_time_ms = (time.perf_counter() - t_start) * 1000
        avg_chunk_ms = http_time_ms / total_chunks if total_chunks > 0 else 0.0

        return RerankBatchResult(
            reranked_patents=reranked_patents,
            total_candidates=len(reranked_patents),
            total_chunks_reranked=total_chunks,
            total_requests=total_requests,
            truncated_chunks_count=truncated_count,
            reranking_query=reranking_query,
            timings={
                "reranker_http_ms": round(http_time_ms, 2),
                "total_ms": round(total_time_ms, 2),
                "avg_per_chunk_ms": round(avg_chunk_ms, 2),
            },
        )

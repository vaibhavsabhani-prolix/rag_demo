"""
Phase 4: Bounded Evidence Retrieval

Given metadata-qualified candidate patents (from Phase 3) and ParsedQuery (from Phase 1),
retrieves a small, bounded, relevant set of patent chunks (with text) for relationship verification.

Key characteristics:
- Evidence query is the LLM-provided parsed_query.evidence_query from Phase 1 - no
  construction here.
- Bounded candidate restriction: Searches only within qualified Phase 3 patent IDs.
- Single batched query embedding (no extra LLM call here, no BGE reranker).
- Zero N+1 Qdrant queries (batched vector search + batched neighbor scroll).
- No per-patent chunk cap - every matched chunk and its neighbors are kept,
  bounded only by EVIDENCE_GLOBAL_TOP_K_CHUNKS across the whole batch.
- Deduplicates chunks per patent.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.config import (
    EVIDENCE_GLOBAL_TOP_K_CHUNKS,
    EVIDENCE_NEIGHBOR_CHUNKS,
)
from app.embedder import Embedder
from app.models.candidate import CandidatePatent
from app.models.collection import SearchCollection
from app.models.evidence import (
    EvidenceChunk,
    EvidenceRetrievalResult,
    PatentEvidence,
)
from app.models.parsed_query import ParsedQuery
from app.qdrant_db import QdrantDB

logger = logging.getLogger(__name__)


def build_evidence_query(parsed_query: ParsedQuery) -> str:
    evidence_query = (parsed_query.evidence_query or "").strip()
    if evidence_query:
        return evidence_query
    return (parsed_query.semantic_query or parsed_query.original_query or "").strip()


class EvidenceRetriever:
    """
    Phase 4 Bounded Evidence Retrieval Engine.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        db: Optional[QdrantDB] = None,
        neighbor_chunks: int = EVIDENCE_NEIGHBOR_CHUNKS,
        global_top_k: int = EVIDENCE_GLOBAL_TOP_K_CHUNKS,
    ):
        self.embedder = embedder or Embedder()
        self.db = db or QdrantDB()
        self.neighbor_chunks = neighbor_chunks
        self.global_top_k = global_top_k

    def retrieve_evidence(
        self,
        parsed_query: ParsedQuery,
        candidates: List[CandidatePatent],
        collection: SearchCollection,
    ) -> EvidenceRetrievalResult:
        """
        Execute Phase 4 Bounded Evidence Retrieval for qualified candidates,
        reading chunks from *collection*.
        """
        t_start = time.perf_counter()

        if not candidates:
            total_time_ms = (time.perf_counter() - t_start) * 1000
            return EvidenceRetrievalResult(
                evidence_by_patent={},
                patent_evidence_list=[],
                total_candidates=0,
                total_evidence_chunks=0,
                evidence_query_text="",
                timings={"total_ms": round(total_time_ms, 2)},
            )

        candidate_patent_ids = [c.patent_id for c in candidates if c.patent_id]
        evidence_query_text = build_evidence_query(parsed_query)

        # Dictionary to accumulate EvidenceChunks per patent: patent_id -> {chunk_id: EvidenceChunk}
        evidence_pool: Dict[str, Dict[int, EvidenceChunk]] = {
            pid: {} for pid in candidate_patent_ids
        }

        embedding_time_ms = 0.0
        qdrant_time_ms = 0.0

        # Step 1: If not metadata-only, perform vector retrieval restricted to candidate patent IDs
        if not parsed_query.is_metadata_only and evidence_query_text:
            # Generate embedding in a single call
            t_embed_start = time.perf_counter()
            vecs = self.embedder.embed_texts([evidence_query_text])
            evidence_vec = vecs[0] if vecs else None
            embedding_time_ms = (time.perf_counter() - t_embed_start) * 1000

            if evidence_vec:
                t_qdrant_start = time.perf_counter()
                candidate_filter = Filter(
                    must=[
                        FieldCondition(
                            key="patent_id",
                            match=MatchAny(any=candidate_patent_ids),
                        )
                    ]
                )

                # No per-patent multiplier here anymore - fetch up to the
                # global cap regardless of candidate count, since there's
                # no per-patent truncation downstream to size this against.
                search_limit = self.global_top_k

                search_res = self.db.client.query_points(
                    collection_name=collection.chunks_collection,
                    query=evidence_vec,
                    query_filter=candidate_filter,
                    limit=search_limit,
                    with_payload=True,
                )

                hits = search_res.points if hasattr(search_res, "points") else search_res
                for hit in hits:
                    payload = hit.payload or {}
                    pid = payload.get("patent_id")
                    cid = payload.get("chunk_id")
                    text = payload.get("text") or ""
                    score = float(hit.score)

                    if not pid or cid is None or pid not in evidence_pool:
                        continue

                    cid_int = int(cid)
                    existing = evidence_pool[pid].get(cid_int)
                    if existing:
                        if score > existing.retrieval_score:
                            existing.retrieval_score = score
                    else:
                        evidence_pool[pid][cid_int] = EvidenceChunk(
                            patent_id=pid,
                            chunk_id=cid_int,
                            text=text,
                            retrieval_score=score,
                            retrieval_source="evidence_query",
                            section=payload.get("section"),
                            document_chunk_index=payload.get("document_chunk_index"),
                            token_count=payload.get("token_count"),
                        )

                qdrant_time_ms = (time.perf_counter() - t_qdrant_start) * 1000

        # Step 2: Identify and fetch bounded neighbor chunks in a single batched query
        t_neighbor_start = time.perf_counter()
        needed_neighbors: Dict[str, Set[int]] = {}

        if self.neighbor_chunks > 0:
            for pid, chunk_map in evidence_pool.items():
                if not chunk_map:
                    continue
                # For each top chunk, target adjacent neighbor indices
                for cid in list(chunk_map.keys()):
                    for offset in range(1, self.neighbor_chunks + 1):
                        prev_cid = cid - offset
                        next_cid = cid + offset
                        if prev_cid >= 0 and prev_cid not in chunk_map:
                            needed_neighbors.setdefault(pid, set()).add(prev_cid)
                        if next_cid not in chunk_map:
                            needed_neighbors.setdefault(pid, set()).add(next_cid)

        if needed_neighbors:
            fetched_neighbors = self._fetch_neighbor_chunks_batch(
                needed_neighbors, collection.chunks_collection
            )
            for (pid, cid), payload in fetched_neighbors.items():
                if pid in evidence_pool and cid not in evidence_pool[pid]:
                    # Find base score of the closest anchor chunk
                    anchor_scores = [
                        c.retrieval_score
                        for c in evidence_pool[pid].values()
                        if abs(c.chunk_id - cid) <= self.neighbor_chunks
                    ]
                    base_score = max(anchor_scores) if anchor_scores else 0.5
                    neighbor_score = round(base_score * 0.95, 4)

                    evidence_pool[pid][cid] = EvidenceChunk(
                        patent_id=pid,
                        chunk_id=cid,
                        text=payload.get("text") or "",
                        retrieval_score=neighbor_score,
                        retrieval_source="neighbor",
                        section=payload.get("section"),
                        document_chunk_index=payload.get("document_chunk_index"),
                        token_count=payload.get("token_count"),
                    )

        neighbor_time_ms = (time.perf_counter() - t_neighbor_start) * 1000

        # Step 3: Deduplicate, sort, and enforce per-patent bounding
        t_dedup_start = time.perf_counter()
        evidence_by_patent: Dict[str, List[EvidenceChunk]] = {}
        patent_evidence_list: List[PatentEvidence] = []
        total_chunks_collected = 0

        # Maintain original candidate order
        for cand in candidates:
            pid = cand.patent_id
            raw_chunks = list(evidence_pool.get(pid, {}).values())

            # Sort chunks: direct evidence_query hits by score desc, then neighbors.
            # No per-patent truncation - every matched chunk and its neighbors are kept.
            def _sort_key(c: EvidenceChunk) -> Tuple[int, float]:
                priority = 0 if c.retrieval_source == "evidence_query" else 1
                return (priority, -c.retrieval_score)

            raw_chunks.sort(key=_sort_key)
            all_chunks = raw_chunks

            evidence_by_patent[pid] = all_chunks
            patent_evidence_list.append(
                PatentEvidence(
                    patent_id=pid,
                    chunks=all_chunks,
                    metadata=cand.metadata,
                    candidate_score=cand.retrieval_score,
                )
            )
            total_chunks_collected += len(all_chunks)

        dedup_time_ms = (time.perf_counter() - t_dedup_start) * 1000
        total_time_ms = (time.perf_counter() - t_start) * 1000

        return EvidenceRetrievalResult(
            evidence_by_patent=evidence_by_patent,
            patent_evidence_list=patent_evidence_list,
            total_candidates=len(candidates),
            total_evidence_chunks=total_chunks_collected,
            evidence_query_text=evidence_query_text,
            timings={
                "embedding_ms": round(embedding_time_ms, 2),
                "qdrant_retrieval_ms": round(qdrant_time_ms, 2),
                "neighbor_ms": round(neighbor_time_ms, 2),
                "deduplication_ms": round(dedup_time_ms, 2),
                "total_ms": round(total_time_ms, 2),
            },
        )

    def _fetch_neighbor_chunks_batch(
        self,
        needed_neighbors: Dict[str, Set[int]],
        chunks_collection: str,
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Fetch neighbor chunk payloads in a single batched Qdrant scroll query.
        """
        target_patent_ids = list(needed_neighbors.keys())
        target_chunk_ids = list({cid for cids in needed_neighbors.values() for cid in cids})

        if not target_patent_ids or not target_chunk_ids:
            return {}

        neighbor_filter = Filter(
            must=[
                FieldCondition(key="patent_id", match=MatchAny(any=target_patent_ids)),
                FieldCondition(key="chunk_id", match=MatchAny(any=target_chunk_ids)),
            ]
        )

        scroll_limit = min(
            len(target_patent_ids) * len(target_chunk_ids),
            500,
        )

        try:
            scroll_res, _ = self.db.client.scroll(
                collection_name=chunks_collection,
                scroll_filter=neighbor_filter,
                limit=scroll_limit,
                with_payload=True,
                with_vectors=False,
            )

            results: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for pt in scroll_res:
                payload = pt.payload or {}
                pid = payload.get("patent_id")
                cid = payload.get("chunk_id")
                if pid and cid is not None:
                    cid_int = int(cid)
                    if pid in needed_neighbors and cid_int in needed_neighbors[pid]:
                        results[(pid, cid_int)] = payload

            return results
        except Exception as e:
            logger.warning("Error scrolling neighbor chunks from Qdrant: %s", e)
            return {}

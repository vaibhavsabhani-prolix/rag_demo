"""
Dynamic Candidate Retriever (Phase 2)

Orchestrates dynamic candidate retrieval from Qdrant:
1. Builds the retrieval views (original / semantic / structured)
2. Generates batched query embeddings
3. Searches each view's chunks in Qdrant
4. Deduplicates chunks and groups them into candidate patents
5. Returns a bounded, grouped candidate pool
"""

import time
import logging
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter

from app.config import RETRIEVAL_TOP_K_CHUNKS
from app.embedder import Embedder
from app.models.candidate import (
    CandidateChunk,
    CandidatePatent,
    CandidateRetrievalResult,
)
from app.models.collection import SearchCollection
from app.models.parsed_query import ParsedQuery
from app.qdrant_db import QdrantDB
from app.retrieval.filter_builder import (
    build_qdrant_filter,
    match_patent_metadata,
)
from app.retrieval.views import build_retrieval_views

logger = logging.getLogger(__name__)


class CandidateRetriever:
    """
    Phase 2 Dynamic Candidate Retrieval Engine.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        db: Optional[QdrantDB] = None,
        candidate_top_k: int = RETRIEVAL_TOP_K_CHUNKS,
    ):
        self.embedder = embedder or Embedder()
        self.db = db or QdrantDB()
        self.candidate_top_k = candidate_top_k

    def retrieve_candidates(
        self, parsed_query: ParsedQuery, collection: SearchCollection
    ) -> CandidateRetrievalResult:

        t_start = time.perf_counter()

        # Branch 1: Metadata-only query (Skip vector search & embedding)
        if parsed_query.is_metadata_only:
            return self._retrieve_metadata_only(parsed_query, collection, t_start)

        # Branch 2: Semantic (+ optional metadata) retrieval
        return self._retrieve_semantic_candidates(parsed_query, collection, t_start)

    def _retrieve_metadata_only(
        self, parsed_query: ParsedQuery, collection: SearchCollection, t_start: float
    ) -> CandidateRetrievalResult:
        """
        Retrieve candidate patents directly matching metadata filters without vector search.
        """
        t_q_start = time.perf_counter()

        q_filter = build_qdrant_filter(parsed_query.metadata_filters)
        matching_patents = self._fetch_matching_patents(
            q_filter, parsed_query.metadata_filters, collection
        )

        qdrant_time_ms = (time.perf_counter() - t_q_start) * 1000

        t_m_start = time.perf_counter()
        candidates: List[CandidatePatent] = []
        for p in matching_patents[: self.candidate_top_k]:
            pid = p.get("patent_id")
            if not pid:
                continue
            candidates.append(
                CandidatePatent(
                    patent_id=pid,
                    best_chunk_id=None,
                    retrieval_score=1.0,
                    matched_views=["metadata_only"],
                    chunk_count=0,
                    chunks=[],
                    metadata=p.get("metadata", {}),
                )
            )

        merge_time_ms = (time.perf_counter() - t_m_start) * 1000
        total_time_ms = (time.perf_counter() - t_start) * 1000

        views = {"metadata_filter": "Metadata-only query constraints"}

        return CandidateRetrievalResult(
            candidates=candidates,
            retrieval_views=views,
            total_chunk_hits=0,
            unique_patents=len(candidates),
            is_metadata_only=True,
            timings={
                "embedding_ms": 0.0,
                "qdrant_retrieval_ms": round(qdrant_time_ms, 2),
                "merge_ms": round(merge_time_ms, 2),
                "total_ms": round(total_time_ms, 2),
            },
        )

    def _retrieve_semantic_candidates(
        self, parsed_query: ParsedQuery, collection: SearchCollection, t_start: float
    ) -> CandidateRetrievalResult:

        # Step 1: Build dynamic retrieval views
        views = build_retrieval_views(parsed_query)
        if not views:
            views = {"original": (parsed_query.original_query or "").strip()}

        # Step 2: Batch embed all unique retrieval views in a SINGLE request
        t_embed_start = time.perf_counter()
        unique_texts = list(dict.fromkeys(views.values()))
        embeddings_list = self.embedder.embed_texts(unique_texts)
        text_to_vec = dict(zip(unique_texts, embeddings_list))
        embedding_time_ms = (time.perf_counter() - t_embed_start) * 1000

        # Step 3: Search each view, collect chunk hits
        t_qdrant_start = time.perf_counter()
        raw_results_per_view: List[tuple[str, Any]] = []
        total_raw_hits = 0

        for view_name, view_text in views.items():
            vec = text_to_vec.get(view_text)
            if not vec:
                continue

            search_res = self.db.client.query_points(
                collection_name=collection.chunks_collection,
                query=vec,
                limit=self.candidate_top_k,
                with_payload=True,
            )

            hits = search_res.points if hasattr(search_res, "points") else search_res
            total_raw_hits += len(hits)
            for hit in hits:
                raw_results_per_view.append((view_name, hit))

        qdrant_time_ms = (time.perf_counter() - t_qdrant_start) * 1000

        # Step 4: Deduplicate and combine chunk candidates
        t_merge_start = time.perf_counter()
        chunk_pool: Dict[str, CandidateChunk] = {}

        for view_name, hit in raw_results_per_view:
            point_id = str(hit.id)
            payload = hit.payload or {}
            patent_id = payload.get("patent_id", "")
            chunk_id = int(payload.get("chunk_id", 0))
            score = float(hit.score)

            if not patent_id:
                continue

            chunk_key = point_id
            if chunk_key in chunk_pool:
                existing = chunk_pool[chunk_key]
                if view_name not in existing.matched_views:
                    existing.matched_views.append(view_name)
                if score > existing.score:
                    existing.score = score
            else:
                chunk_pool[chunk_key] = CandidateChunk(
                    point_id=point_id,
                    patent_id=patent_id,
                    chunk_id=chunk_id,
                    score=score,
                    matched_views=[view_name],
                    section=payload.get("section"),
                    text=payload.get("text"),
                    document_chunk_index=payload.get("document_chunk_index"),
                    token_count=payload.get("token_count"),
                )

        # Step 5: Group chunks by patent_id
        patent_chunks_map: Dict[str, List[CandidateChunk]] = {}
        for chunk in chunk_pool.values():
            patent_chunks_map.setdefault(chunk.patent_id, []).append(chunk)

        # Build candidate patents
        candidate_patents: List[CandidatePatent] = []
        for patent_id, chunks in patent_chunks_map.items():
            best_chunk = max(chunks, key=lambda c: c.score)
            all_views = sorted(list(set(v for c in chunks for v in c.matched_views)))
            candidate_patents.append(
                CandidatePatent(
                    patent_id=patent_id,
                    best_chunk_id=best_chunk.chunk_id,
                    retrieval_score=round(best_chunk.score, 4),
                    matched_views=all_views,
                    chunk_count=len(chunks),
                    chunks=chunks,
                )
            )

        # Order by strongest retrieval evidence
        candidate_patents.sort(key=lambda p: p.retrieval_score, reverse=True)

        # Step 6: Enrich with patent metadata
        top_patent_ids = [p.patent_id for p in candidate_patents]
        meta_dict = self.db.get_patents_metadata(top_patent_ids, collection.patents_collection)
        for p in candidate_patents:
            p.metadata = meta_dict.get(p.patent_id, {})

        merge_time_ms = (time.perf_counter() - t_merge_start) * 1000
        total_time_ms = (time.perf_counter() - t_start) * 1000

        return CandidateRetrievalResult(
            candidates=candidate_patents,
            retrieval_views=views,
            total_chunk_hits=total_raw_hits,
            unique_patents=len(candidate_patents),
            is_metadata_only=False,
            timings={
                "embedding_ms": round(embedding_time_ms, 2),
                "qdrant_retrieval_ms": round(qdrant_time_ms, 2),
                "merge_ms": round(merge_time_ms, 2),
                "total_ms": round(total_time_ms, 2),
            },
        )

    def _fetch_matching_patents(
        self,
        q_filter: Optional[Filter],
        metadata_filters: list,
        collection: SearchCollection,
    ) -> List[Dict[str, Any]]:
        """
        Scroll patent metadata collection matching Qdrant filter and verify with Python logic.
        """
        matching: List[Dict[str, Any]] = []

        try:
            # Scroll with filter from patents metadata collection
            results, _ = self.db.client.scroll(
                collection_name=collection.patents_collection,
                scroll_filter=q_filter,
                limit=1000,
                with_payload=True,
            )
            for pt in results:
                payload = pt.payload or {}
                meta = payload.get("metadata", {})
                if match_patent_metadata(meta, metadata_filters):
                    matching.append(payload)
        except Exception as exc:
            logger.warning("Qdrant metadata filtered scroll failed, falling back to full scan: %s", exc)
            # Fallback: scan metadata points and evaluate in Python
            try:
                results, _ = self.db.client.scroll(
                    collection_name=collection.patents_collection,
                    limit=2000,
                    with_payload=True,
                )
                for pt in results:
                    payload = pt.payload or {}
                    meta = payload.get("metadata", {})
                    if match_patent_metadata(meta, metadata_filters):
                        matching.append(payload)
            except Exception as e:
                logger.error("Failed to scroll patent metadata: %s", e)

        return matching

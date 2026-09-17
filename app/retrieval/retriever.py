"""
Dynamic Candidate Retriever (Phase 2)

Orchestrates dynamic candidate retrieval from Qdrant:
1. Builds dynamic semantic and structured retrieval views
2. Generates batched query embeddings
3. Executes Qdrant vector retrieval with dynamic metadata filters
4. Combines and deduplicates chunks into candidate patents
5. Returns a bounded, grouped candidate pool
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.config import (
    CHUNKS_COLLECTION_NAME,
    PATENTS_COLLECTION_NAME,
    PATENT_CANDIDATE_TOP_K,
    RETRIEVAL_TOP_K_PER_VIEW,
)
from app.embedder import Embedder
from app.models.candidate import (
    CandidateChunk,
    CandidatePatent,
    CandidateRetrievalResult,
)
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
        top_k_per_view: int = RETRIEVAL_TOP_K_PER_VIEW,
        candidate_top_k: int = PATENT_CANDIDATE_TOP_K,
    ):
        self.embedder = embedder or Embedder()
        self.db = db or QdrantDB()
        self.top_k_per_view = top_k_per_view
        self.candidate_top_k = candidate_top_k

    def retrieve_candidates(self, parsed_query: ParsedQuery) -> CandidateRetrievalResult:
        """
        Execute Phase 2 Candidate Retrieval for a ParsedQuery.
        """
        t_start = time.perf_counter()

        # Branch 1: Metadata-only query (Skip vector search & embedding)
        if parsed_query.is_metadata_only:
            return self._retrieve_metadata_only(parsed_query, t_start)

        # Branch 2: Semantic (+ optional metadata) retrieval
        return self._retrieve_semantic_candidates(parsed_query, t_start)

    def _retrieve_metadata_only(
        self, parsed_query: ParsedQuery, t_start: float
    ) -> CandidateRetrievalResult:
        """
        Retrieve candidate patents directly matching metadata filters without vector search.
        """
        t_q_start = time.perf_counter()

        q_filter = build_qdrant_filter(parsed_query.metadata_filters)
        matching_patents = self._fetch_matching_patents(q_filter, parsed_query.metadata_filters)

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
        self, parsed_query: ParsedQuery, t_start: float
    ) -> CandidateRetrievalResult:
        """
        Execute multi-view semantic candidate retrieval with dynamic metadata pre-filtering.
        """
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

        # Step 3: Handle metadata filter if present
        t_qdrant_start = time.perf_counter()
        chunk_filter: Optional[Filter] = None

        if parsed_query.metadata_filters:
            meta_filter = build_qdrant_filter(parsed_query.metadata_filters)
            matching_patents = self._fetch_matching_patents(
                meta_filter, parsed_query.metadata_filters
            )
            matching_patent_ids = [p["patent_id"] for p in matching_patents if "patent_id" in p]

            if not matching_patent_ids:
                # The metadata pre-filter is an optimization (narrow the
                # vector search to patents we already know match, which is
                # both faster and more precise than pure semantic top-K
                # search) - it is NOT meant to be a hard gate. If it finds
                # zero patents, that's just as likely to be an extraction/
                # mapping gap (wrong field code, formatting mismatch, an
                # LLM-invented constraint) as a genuine "no such patent."
                # Rather than killing the search outright, fall back to an
                # unrestricted vector search across the whole collection
                # and let Phase 3's metadata_filter (which runs on whatever
                # candidates come back) do the real enforcement - the same
                # place every other candidate already gets checked.
                print(
                    f"[Retriever] Metadata pre-filter matched 0 patents for "
                    f"{len(parsed_query.metadata_filters)} filter(s); falling back to "
                    f"unrestricted vector search, Phase 3 will still enforce the filters"
                )
                chunk_filter = None
            else:
                chunk_filter = Filter(
                    must=[
                        FieldCondition(
                            key="patent_id",
                            match=MatchAny(any=matching_patent_ids),
                        )
                    ]
                )

        # Step 4: Retrieve candidate chunks per view
        raw_results_per_view: List[tuple[str, Any]] = []
        total_raw_hits = 0

        for view_name, view_text in views.items():
            vec = text_to_vec.get(view_text)
            if not vec:
                continue

            search_res = self.db.client.query_points(
                collection_name=CHUNKS_COLLECTION_NAME,
                query=vec,
                query_filter=chunk_filter,
                limit=self.top_k_per_view,
                with_payload=True,
            )

            hits = search_res.points if hasattr(search_res, "points") else search_res
            total_raw_hits += len(hits)
            for hit in hits:
                raw_results_per_view.append((view_name, hit))

        qdrant_time_ms = (time.perf_counter() - t_qdrant_start) * 1000

        # Step 5: Deduplicate and combine chunk candidates
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

        # Step 6: Group chunks by patent_id
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

        # Order by strongest retrieval evidence and bound candidate pool
        candidate_patents.sort(key=lambda p: p.retrieval_score, reverse=True)
        bounded_candidates = candidate_patents[: self.candidate_top_k]

        # Step 7: Enrich with patent metadata
        top_patent_ids = [p.patent_id for p in bounded_candidates]
        meta_dict = self.db.get_patents_metadata(top_patent_ids)
        for p in bounded_candidates:
            p.metadata = meta_dict.get(p.patent_id, {})

        merge_time_ms = (time.perf_counter() - t_merge_start) * 1000
        total_time_ms = (time.perf_counter() - t_start) * 1000

        return CandidateRetrievalResult(
            candidates=bounded_candidates,
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
        self, q_filter: Optional[Filter], metadata_filters: list
    ) -> List[Dict[str, Any]]:
        """
        Scroll patent metadata collection matching Qdrant filter and verify with Python logic.
        """
        matching: List[Dict[str, Any]] = []

        try:
            # Scroll with filter from patents metadata collection
            results, _ = self.db.client.scroll(
                collection_name=PATENTS_COLLECTION_NAME,
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
                    collection_name=PATENTS_COLLECTION_NAME,
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

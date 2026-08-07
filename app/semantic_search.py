"""
Patent-Level Semantic Search

Pipeline:

    Query
        ↓
    Embed Query
        ↓
    Qdrant Vector Search (chunk-level)
        ↓
    Cross-Encoder Reranker (chunk-level)
        ↓
    Group by patent_id
        ↓
    Compute patent score (max reranker score)
        ↓
    Sort patents by score
        ↓
    Return PatentSearchResult list

Internal retrieval unit: Chunk
External retrieval unit: Patent
"""

from __future__ import annotations

from collections import defaultdict

from app.embedder import Embedder
from app.qdrant_db import QdrantDB
from app.reranker import Reranker
from app.models.patent_search_result import PatentSearchResult, RankedChunk


class SemanticSearch:

    def __init__(self):

        self.embedder = Embedder()
        self.db = QdrantDB()
        self.reranker = Reranker()

    def search_detailed(
        self, query: str
    ) -> tuple[list, list[tuple[float, object]], list[PatentSearchResult]]:
        """
        Search for patents and return results at each pipeline stage:
        1. Qdrant vector search candidate chunks (list of ScoredPoint)
        2. Reranked candidate chunks (list of (reranker_score, ScoredPoint))
        3. Aggregated PatentSearchResult list
        """
        # Step 1: Embed the query
        query_vector = self.embedder.embed_query(query)

        # Step 2: Retrieve candidate chunks from Qdrant
        qdrant_results = self.db.search(
            query_vector=query_vector,
        )

        # Step 3: Rerank the candidate chunks
        reranked_results = self.reranker.rerank(
            query=query,
            results=qdrant_results,
        )

        # Step 4: Aggregate chunks into patent-level results
        patent_results = self._aggregate_by_patent(reranked_results)

        return qdrant_results, reranked_results, patent_results

    def search(self, query: str) -> list[PatentSearchResult]:
        """
        Search for patents matching *query*.

        Returns one PatentSearchResult per patent, sorted by
        patent-level score (highest reranker score among chunks).
        """
        _, _, patent_results = self.search_detailed(query)
        return patent_results

    # ==============================================================
    # Patent aggregation
    # ==============================================================

    @staticmethod
    def _aggregate_by_patent(
        reranked: list[tuple[float, object]],
    ) -> list[PatentSearchResult]:
        """
        Group reranked chunks by patent_id and produce one
        PatentSearchResult per patent.

        Patent score = highest reranker score among all chunks.
        """

        if not reranked:
            return []

        # ---- Group chunks by patent_id ----
        patent_chunks: dict[str, list[RankedChunk]] = defaultdict(list)
        patent_metadata: dict[str, dict] = {}

        for score, result in reranked:

            payload = result.payload
            patent_id = payload["patent_id"]

            ranked_chunk = RankedChunk(
                chunk_id=payload.get("chunk_id", 0),
                section=payload.get("section", ""),
                text=payload.get("text", ""),
                score=score,
                token_count=payload.get("token_count", 0),
                word_count=payload.get("word_count", 0),
                section_chunk_index=payload.get("section_chunk_index", 0),
                document_chunk_index=payload.get("document_chunk_index", 0),
                total_chunks=payload.get("total_chunks", 0),
                start_offset=payload.get("start_offset", 0),
                end_offset=payload.get("end_offset", 0),
            )

            patent_chunks[patent_id].append(ranked_chunk)

            # Store metadata once per patent (first seen)
            if patent_id not in patent_metadata:
                patent_metadata[patent_id] = payload.get("metadata", {})

        # ---- Build PatentSearchResult per patent ----
        patent_results: list[PatentSearchResult] = []

        for patent_id, chunks in patent_chunks.items():

            # Sort chunks by reranker score descending
            chunks.sort(key=lambda c: c.score, reverse=True)

            best = chunks[0]

            patent_results.append(
                PatentSearchResult(
                    patent_id=patent_id,
                    score=best.score,
                    best_chunk=best,
                    matching_chunks=chunks,
                    metadata=patent_metadata[patent_id],
                )
            )

        # ---- Sort patents by score descending ----
        patent_results.sort(key=lambda p: p.score, reverse=True)

        return patent_results
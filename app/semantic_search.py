"""
Patent-Level Semantic Search

Pipeline:

    Query
        ↓
    Query Understanding -> semantic_query + metadata_filters
        ↓
    Embed semantic_query ONLY
        ↓
    Qdrant Vector Search on patent_chunks, GROUPED by patent_id
    (pure semantic search, no metadata filter involved - identifies
    the top PATENT_CANDIDATE_TOP_K distinct PATENTS, returning each
    patent's top CANDIDATE_CHUNKS_PER_PATENT best-matching chunks)
        ↓
    Fetch metadata for ONLY those candidate patent_ids
    ("patents" collection, via the existing get_patents_metadata())
        ↓
    [metadata_filters present?]
        │ yes                                   │ no
        ▼                                       │
    Evaluate FilterEngine.matches() per         │
    candidate patent; keep only chunks          │
    belonging to a patent that matches          │
        ▼                                       ▼
    Cross-Encoder Reranker (semantic_query only - metadata
    phrases were already split off by Query Understanding;
    scores every surviving chunk, no truncation yet)
                          ↓
                   Group by patent_id
                          ↓
        Fetch metadata again for the FINAL (small,
        post-rerank) patent set, for PatentSearchResult display
                          ↓
        Compute patent score (max reranker score)
                          ↓
                   Sort patents by score
                          ↓
              Truncate to FINAL_TOP_K patents
                          ↓
              Return PatentSearchResult list

Internal retrieval unit: Chunk
External retrieval unit: Patent

Metadata filtering happens AFTER semantic vector search, on the
candidate chunks that search already returned - never before it, and
never as a Qdrant-side filter on the search itself (QdrantDB.search()
is pure vector search, no patent_ids parameter). The "patents"
collection is only ever consulted for patent_ids that are already
semantically relevant candidates. See app/query_understanding/ for how
natural language becomes semantic_query + metadata_filters, and
app/filter_engine.py for how a MetadataFilter is evaluated against a
patent's metadata dict.

QdrantDB.search()'s group-by search returns each candidate patent's top
CANDIDATE_CHUNKS_PER_PATENT chunks, then flattens every patent's group
of hits into one flat list - so the same patent_id can legitimately
appear multiple times in that raw list. That chunk-level pool is kept
intact internally (as `candidate_chunks` in search_detailed) because
metadata filtering and the reranker are designed to see every chunk of
a candidate patent, not just one. The `qdrant_results` returned by
search_detailed() and shown in the UI as "Qdrant Vector Search
Candidates" is a separate, patent-level view derived from that same
pool - see _dedupe_top_chunk_per_patent() - with at most one entry per
patent_id (that patent's single highest-scoring chunk), so the
displayed candidate count reflects distinct patents, not duplicated
chunks.

Nothing about ingestion, chunking, embedding, or the patent_chunks/
patents collection schemas changes here - this module only reorders how
the existing pieces (QdrantDB.search, QdrantDB.get_patents_metadata,
Reranker, aggregation) are called.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from app.config import FINAL_TOP_K
from app.embedder import Embedder
from app.filter_engine import FilterEngine
from app.models.patent_search_result import PatentSearchResult, RankedChunk, ScoreBreakdown
from app.qdrant_db import QdrantDB
from app.query_understanding import ParsedQuery, QueryUnderstanding
from app.reranker import Reranker


def _dedupe_top_chunk_per_patent(chunks: list) -> list:
    """
    Collapse *chunks* to at most one entry per patent_id - that
    patent's single highest-scoring chunk, with its original Qdrant
    score and payload preserved unchanged.

    Duplicate patent_ids exist in *chunks* because QdrantDB.search()'s
    group-by search returns each candidate patent's top
    CANDIDATE_CHUNKS_PER_PATENT chunks and then flattens every patent's
    group of hits into one flat list (see QdrantDB.search) - a patent
    with several similar-scoring chunks can appear more than once in
    that flat list. This only re-collapses that flattening for the
    patent-level candidate view (qdrant_results); it must NOT be
    applied to the chunk-level pool used for metadata filtering/
    reranking, which intentionally keeps multiple chunks per patent.

    *chunks* is expected to already be filtered to points that carry a
    patent_id payload (see search_detailed's candidate_chunks).
    """

    best_by_patent: dict[str, object] = {}
    for chunk in chunks:
        patent_id = chunk.payload["patent_id"]
        current_best = best_by_patent.get(patent_id)
        if current_best is None or chunk.score > current_best.score:
            best_by_patent[patent_id] = chunk

    return sorted(best_by_patent.values(), key=lambda c: c.score, reverse=True)


class SemanticSearch:
    def __init__(self):

        self.embedder = Embedder()
        self.db = QdrantDB()
        self.reranker = Reranker()
        self.query_understanding = QueryUnderstanding()

    def search_detailed(
        self,
        query: str,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> tuple[
        ParsedQuery, list, list, list[tuple[float, object, ScoreBreakdown]], list[PatentSearchResult]
    ]:
        """
        Search for patents and return results at each pipeline stage:
        1. Parsed query (semantic_query + metadata_filters)
        2. Initial Qdrant vector search candidates, one entry per
           unique candidate patent_id - that patent's single
           highest-scoring chunk (list of ScoredPoint; see
           _dedupe_top_chunk_per_patent). The internal pool used by
           steps 3-4 below still holds every chunk (up to
           CANDIDATE_CHUNKS_PER_PATENT per patent) - only this
           returned/displayed list is deduped to patent-level.
        3. Candidate chunks after metadata filtering
        4. Reranked candidate chunks (list of (final_score, ScoredPoint, ScoreBreakdown))
        5. Aggregated PatentSearchResult list

        If *on_stage* is given, it's called after each pipeline stage
        completes with (stage_name, elapsed_seconds) - lets a caller
        (e.g. the UI) report per-stage progress and timing as the
        search runs, rather than only after everything finishes.
        """

        def _run(stage_name, fn, *args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if on_stage is not None:
                on_stage(stage_name, time.perf_counter() - start)
            return result

        # Step 0: Query Understanding
        parsed = _run("Query Understanding", self.query_understanding.parse, query)

        # filter query  only, no semantic query to embed or rerank
        if parsed.is_metadata_only:
            return self._search_by_metadata_only(parsed, on_stage=on_stage)

        # Step 1: Embed the semantic portion only - metadata-filter
        # phrases were already split off by Query Understanding, so
        # neither embedding nor reranking ever sees e.g. "US" or
        # "Coca Cola" as if they were part of the topic being searched.
        query_vector = _run(
            "Embedding", self.embedder.embed_query, parsed.semantic_query
        )

        # Step 2: Pure semantic vector search - no metadata filter here.
        # Identifies the top candidate PATENTS, returning each patent's
        # top CANDIDATE_CHUNKS_PER_PATENT best-matching chunks (Qdrant
        # group-by search group_size, see QdrantDB.search). QdrantDB.search
        # flattens every patent's chunk group into one flat list, so the
        # SAME patent_id can appear multiple times here - that flattening
        # is exactly where duplicate patent_ids enter the pipeline.
        vector_search_chunks = _run(
            "Vector Search", self.db.search, query_vector=query_vector
        )

        # candidate_chunks intentionally stays chunk-level (multiple
        # chunks per patent) - it's the pool metadata filtering and the
        # reranker (steps 3-4 below) actually operate on, and the
        # reranker relies on CANDIDATE_CHUNKS_PER_PATENT giving it more
        # than one chunk per patent to score. Do not dedupe this pool.
        candidate_chunks = [
            point
            for point in vector_search_chunks
            if point.payload and point.payload.get("patent_id")
        ]

        # qdrant_results, in contrast, is the patent-level candidate view
        # returned below and shown in the UI as "Qdrant Vector Search
        # Candidates" - it must contain at most one entry per patent_id.
        # Collapse each patent's chunks down to its single
        # highest-scoring chunk, keeping that chunk's original Qdrant
        # score and payload unchanged (see _dedupe_top_chunk_per_patent).
        # This is a display/reporting-level fix only: candidate_chunks
        # above is untouched and still feeds filtering/reranking with
        # every chunk.
        qdrant_results = _dedupe_top_chunk_per_patent(candidate_chunks)

        # Step 3: Metadata filtering, on the candidates just fetched.
        # Only patents that are already semantic candidates ever get
        # their metadata looked up - never the whole "patents" collection.
        if parsed.metadata_filters:
            filtered_results = _run(
                "Metadata Filtering",
                self._filter_candidates_by_metadata,
                candidate_chunks,
                parsed.metadata_filters,
            )
        else:
            filtered_results = candidate_chunks

        # Step 4: Rerank whatever survived filtering. Passing `parsed`
        # lets the reranker blend in requirement/constraint/relationship
        # coverage on top of the semantic score when Query Understanding
        # extracted a requirements structure (see app/reranker.py) -
        # falls back to pure semantic reranking otherwise.
        reranked_results = _run(
            "Reranking",
            self.reranker.rerank,
            query=parsed.semantic_query,
            results=filtered_results,
            requirements=parsed,
        )

        # Step 5: Aggregate chunks into patent-level results, then keep
        # only the top FINAL_TOP_K patents by patent score. Truncating
        # here (post-aggregation) rather than on the chunk list means a
        # patent survives on its best chunk regardless of how many other
        # patents' chunks outscored its weaker ones.
        patent_results = _run(
            "Aggregation",
            lambda: self._aggregate_by_patent(reranked_results)[:FINAL_TOP_K],
        )

        return (
            parsed,
            qdrant_results,
            filtered_results,
            reranked_results,
            patent_results,
        )

    def search(self, query: str) -> list[PatentSearchResult]:
        """
        Search for patents matching *query*.

        Returns one PatentSearchResult per patent, sorted by
        patent-level score (highest reranker score among chunks).
        """
        _, _, _, _, patent_results = self.search_detailed(query)
        return patent_results

    # ==============================================================
    # Metadata-only search (no topic to embed, vector-search, or rerank)
    # ==============================================================

    def _search_by_metadata_only(
        self,
        parsed: ParsedQuery,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> tuple[
        ParsedQuery, list, list, list[tuple[float, object, ScoreBreakdown]], list[PatentSearchResult]
    ]:
        """
        Handle a query whose semantic_query came back empty (see
        parser.py Rule 9B) - it's purely metadata filters, e.g.
        "applications filed in 2008 by Wyeth".

        Filters the whole "patents" collection directly instead of the
        PATENT_CANDIDATE_TOP_K vector-search candidate pool, fetches every chunk
        belonging to the matching patents, and aggregates them straight
        into PatentSearchResult - no embedding, no vector search, no
        reranker, since there's no query text to score chunks against.
        """

        def _run(stage_name, fn, *args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if on_stage is not None:
                on_stage(stage_name, time.perf_counter() - start)
            return result

        matching_patent_ids = _run(
            "Metadata Filtering", self.db.filter_patent_ids, parsed.metadata_filters
        )

        if not matching_patent_ids:
            return parsed, [], [], [], []

        chunks = _run(
            "Chunk Retrieval", self.db.get_chunks_for_patent_ids, matching_patent_ids
        )

        # chunks already carry a placeholder .score (see
        # QdrantDB.get_chunks_for_patent_ids) so they slot straight into
        # the (score, chunk, breakdown) shape _aggregate_by_patent expects
        # from the reranker, without actually reranking anything.
        reranked_results = [
            (chunk.score, chunk, ScoreBreakdown(semantic_score=chunk.score, final_score=chunk.score))
            for chunk in chunks
        ]

        patent_results = _run(
            "Aggregation", self._aggregate_by_patent, reranked_results
        )

        return parsed, chunks, chunks, reranked_results, patent_results

    # ==============================================================
    # Metadata filtering (post-vector-search, pre-rerank)
    # ==============================================================

    def _filter_candidates_by_metadata(
        self,
        qdrant_results: list,
        metadata_filters: list,
    ) -> list:
        """
        Keep only candidate chunks whose patent satisfies every filter.

        Extracts the unique patent_ids already present in the semantic
        candidates, fetches their metadata in ONE batch call (the
        existing get_patents_metadata - no per-patent requests), and
        evaluates FilterEngine.matches() per patent. A patent with no
        stored metadata can't satisfy any filter and is excluded.
        """

        patent_ids = list(
            dict.fromkeys(
                point.payload.get("patent_id")
                for point in qdrant_results
                if point.payload and point.payload.get("patent_id")
            )
        )

        patent_metadata = self.db.get_patents_metadata(patent_ids)

        matching_ids = {
            patent_id
            for patent_id, metadata in patent_metadata.items()
            if FilterEngine.matches(metadata, metadata_filters)
        }

        return [
            point
            for point in qdrant_results
            if point.payload and point.payload.get("patent_id") in matching_ids
        ]

    # ==============================================================
    # Patent aggregation (unchanged)
    # ==============================================================

    def _aggregate_by_patent(
        self,
        reranked: list[tuple[float, object, ScoreBreakdown]],
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

        for score, result, breakdown in reranked:
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
                breakdown=breakdown,
            )

            patent_chunks[patent_id].append(ranked_chunk)

        # ---- Fetch metadata once per unique patent ----
        patent_metadata = self.db.get_patents_metadata(list(patent_chunks.keys()))

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
                    metadata=patent_metadata.get(patent_id, {}),
                )
            )

        # ---- Sort patents by score descending ----
        patent_results.sort(key=lambda p: p.score, reverse=True)

        return patent_results

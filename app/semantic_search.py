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
    the top PATENT_CANDIDATE_TOP_K distinct PATENTS by their single
    best-matching chunk each)
        ↓
    Extract unique patent_id[] from those candidates
        ↓
    Fetch EVERY chunk belonging to those candidate patents
    (get_chunks_for_patent_ids - unbounded per patent, not a fixed cap)
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

Nothing about ingestion, chunking, embedding, or the patent_chunks/
patents collection schemas changes here - this module only reorders how
the existing pieces (QdrantDB.search, QdrantDB.get_patents_metadata,
Reranker, aggregation) are called.
"""

from __future__ import annotations

from collections import defaultdict

from app.config import FINAL_TOP_K
from app.embedder import Embedder
from app.filter_engine import FilterEngine
from app.models.patent_search_result import PatentSearchResult, RankedChunk
from app.qdrant_db import QdrantDB
from app.query_understanding import ParsedQuery, QueryUnderstanding
from app.reranker import Reranker


class SemanticSearch:

    def __init__(self):

        self.embedder = Embedder()
        self.db = QdrantDB()
        self.reranker = Reranker()
        self.query_understanding = QueryUnderstanding()

    def search_detailed(
        self, query: str
    ) -> tuple[ParsedQuery, list, list, list[tuple[float, object]], list[PatentSearchResult]]:
        """
        Search for patents and return results at each pipeline stage:
        1. Parsed query (semantic_query + metadata_filters)
        2. Initial Qdrant vector search candidates - one best-matching
           chunk per candidate patent (list of ScoredPoint)
        3. Candidate chunks after metadata filtering - now every chunk
           of every surviving candidate patent, not just the one used
           to identify it
        4. Reranked candidate chunks (list of (reranker_score, ScoredPoint))
        5. Aggregated PatentSearchResult list
        """

        # Step 0: Query Understanding
        parsed = self.query_understanding.parse(query)

        # A query that is purely metadata filters with no real topic
        # (e.g. "applications filed in 2008 by Wyeth") comes back from
        # Query Understanding with semantic_query == "" (see parser.py
        # Rule 9B) - there's nothing meaningful to vector-search or
        # rerank, and restricting to the PATENT_CANDIDATE_TOP_K candidate
        # pool would wrongly exclude matching patents never picked up by
        # an embedding of a non-existent topic. Filter the whole
        # collection directly instead.
        if parsed.is_metadata_only:
            return self._search_by_metadata_only(parsed)

        # Step 1: Embed the semantic portion only - metadata-filter
        # phrases were already split off by Query Understanding, so
        # neither embedding nor reranking ever sees e.g. "US" or
        # "Coca Cola" as if they were part of the topic being searched.
        query_vector = self.embedder.embed_query(parsed.semantic_query)

        # Step 2: Pure semantic vector search - no metadata filter here.
        # Identifies the top candidate PATENTS only (one representative
        # best-matching chunk each) - not the chunk pool that gets
        # filtered/reranked next.
        qdrant_results = self.db.search(query_vector=query_vector)

        candidate_patent_ids = list(
            dict.fromkeys(
                point.payload.get("patent_id")
                for point in qdrant_results
                if point.payload and point.payload.get("patent_id")
            )
        )

        # Fetch every chunk belonging to those candidate patents - each
        # patent's full chunk set, unbounded, not an arbitrary fixed cap
        # per patent. This is what actually gets filtered and reranked.
        candidate_chunks = self.db.get_chunks_for_patent_ids(candidate_patent_ids)

        # Step 3: Metadata filtering, on the candidates just fetched.
        # Only patents that are already semantic candidates ever get
        # their metadata looked up - never the whole "patents" collection.
        if parsed.metadata_filters:
            filtered_results = self._filter_candidates_by_metadata(
                candidate_chunks,
                parsed.metadata_filters,
            )
        else:
            filtered_results = candidate_chunks

        # Step 4: Rerank whatever survived filtering.
        reranked_results = self.reranker.rerank(
            query=parsed.semantic_query,
            results=filtered_results,
        )

        # Step 5: Aggregate chunks into patent-level results, then keep
        # only the top FINAL_TOP_K patents by patent score. Truncating
        # here (post-aggregation) rather than on the chunk list means a
        # patent survives on its best chunk regardless of how many other
        # patents' chunks outscored its weaker ones.
        patent_results = self._aggregate_by_patent(reranked_results)[:FINAL_TOP_K]

        return parsed, qdrant_results, filtered_results, reranked_results, patent_results

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
        self, parsed: ParsedQuery
    ) -> tuple[ParsedQuery, list, list, list[tuple[float, object]], list[PatentSearchResult]]:
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

        matching_patent_ids = self.db.filter_patent_ids(parsed.metadata_filters)

        if not matching_patent_ids:
            return parsed, [], [], [], []

        chunks = self.db.get_chunks_for_patent_ids(matching_patent_ids)

        # chunks already carry a placeholder .score (see
        # QdrantDB.get_chunks_for_patent_ids) so they slot straight into
        # the (score, chunk) shape _aggregate_by_patent expects from the
        # reranker, without actually reranking anything.
        reranked_results = [(chunk.score, chunk) for chunk in chunks]

        patent_results = self._aggregate_by_patent(reranked_results)

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

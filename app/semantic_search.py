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
    the top PATENT_CANDIDATE_TOP_K distinct candidate PATENTS; this
    step is only for deciding WHICH patents are worth considering, not
    for deciding which of their chunks are - see below)
        ↓
    [metadata_filters present?]
        │ yes                                   │ no
        ▼                                       │
    Evaluate FilterEngine.matches() per         │
    candidate patent_id (metadata lookup only,  │
    no chunk text needed yet)                   │
        ▼                                       ▼
    Fetch EVERY indexed chunk of each surviving candidate patent
    (QdrantDB.get_chunks_for_patent_ids - unbounded, not just the
    handful the initial vector search happened to surface)
                          ↓
    Cross-Encoder Reranker (semantic_query only - metadata
    phrases were already split off by Query Understanding) scores
    EVERY one of those chunks individually; a patent's score is the
    MAX across its own chunks (see app/reranker.py) - checking the
    patent's complete text, not a similarity-biased subset.
                          ↓
    Keep only patents scoring >= PATENT_RELEVANCE_THRESHOLD
                          ↓
                   Group by patent_id
                          ↓
        Fetch metadata again for the FINAL (small,
        post-rerank) patent set, for PatentSearchResult display
                          ↓
             Sort (question queries: a patent with a
             genuine extracted answer leads; otherwise
             by relevance score)
                          ↓
              Truncate to FINAL_TOP_K patents
                          ↓
              Return PatentSearchResult list

Internal retrieval unit: Chunk
External retrieval unit: Patent

Metadata filtering happens AFTER the vector search that identifies
candidate PATENTS, but BEFORE their chunks are fetched for reranking -
never as a Qdrant-side filter on the vector search itself
(QdrantDB.search() is pure vector search, no patent_ids parameter), and
never against the whole "patents" collection (only candidate patent_ids
that are already semantically relevant ever get their metadata looked
up). See app/query_understanding/ for how natural language becomes
semantic_query + metadata_filters, and app/filter_engine.py for how a
MetadataFilter is evaluated against a patent's metadata dict.

QdrantDB.search()'s group-by search returns each candidate patent's top
CANDIDATE_CHUNKS_PER_PATENT chunks purely to identify WHICH patents are
worth considering (a cheap embedding-similarity signal) - it is NOT
what gets reranked. Once metadata filtering narrows the candidate
patent_ids, search_detailed() fetches every chunk each surviving
patent actually has (QdrantDB.get_chunks_for_patent_ids, unbounded) and
hands that complete set to the reranker, so a patent's relevance is
decided by its strongest chunk out of everything it has, not out of
whichever handful vector search's embedding pass happened to surface.
The `qdrant_results` returned by search_detailed() and shown in the UI
as "Qdrant Vector Search Candidates" is a separate, patent-level view
derived from the initial vector-search chunks only - see
_dedupe_top_chunk_per_patent() - with at most one entry per patent_id,
so the displayed candidate count reflects distinct patents, not
duplicated chunks; it does not reflect the fuller chunk set reranking
actually sees.

Nothing about ingestion, chunking, embedding, or the patent_chunks/
patents collection schemas changes here - this module only reorders how
the existing pieces (QdrantDB.search, QdrantDB.get_chunks_for_patent_ids,
QdrantDB.get_patents_metadata, Reranker, aggregation) are called.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from app.config import FINAL_TOP_K, PATENT_RELEVANCE_THRESHOLD
from app.embedder import Embedder
from app.evidence_selector import EvidenceSelector
from app.filter_engine import FilterEngine
from app.models.patent_search_result import PatentSearchResult, RankedChunk
from app.qdrant_db import QdrantDB
from app.query_understanding import ParsedQuery, QueryUnderstanding
from app.reranker import Reranker


def _dedupe_top_chunk_per_patent(chunks: list) -> list:
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
        self.db.ensure_payload_index()
        self.reranker = Reranker()
        self.query_understanding = QueryUnderstanding()
        self.evidence_selector = EvidenceSelector()

    def search_detailed(
        self,
        query: str,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> tuple[
        ParsedQuery,
        list,
        list,
        list[tuple[float, object]],
        list[PatentSearchResult],
    ]:
        def _run(stage_name, fn, *args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if on_stage is not None:
                on_stage(stage_name, time.perf_counter() - start)
            return result

        #  Query Understanding
        parsed = _run("Query Understanding", self.query_understanding.parse, query)

        return self.search_from_parsed(parsed, on_stage=on_stage)

    def search_from_parsed(
        self,
        parsed: ParsedQuery,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> tuple[
        ParsedQuery,
        list,
        list,
        list[tuple[float, object]],
        list[PatentSearchResult],
    ]:
        """
        Everything search_detailed() does AFTER Query Understanding - split
        out so a caller that already has a ParsedQuery (e.g. the UI, after
        the user picked one specific field out of an UNCERTAIN FIELD OR
        group - see MetadataFilter.group) can run the rest of the pipeline
        without re-invoking the LLM.
        """

        def _run(stage_name, fn, *args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if on_stage is not None:
                on_stage(stage_name, time.perf_counter() - start)
            return result

        # filter query only, no semantic query to embed or rerank
        if parsed.is_metadata_only:
            return self._search_by_metadata_only(parsed, on_stage=on_stage)

        # convert query into embedding vector
        query_vector = _run(
            "Embedding", self.embedder.embed_query, parsed.semantic_query
        )

        # search from qdrantDB return the chunks
        vector_search_chunks = _run(
            "Vector Search", self.db.search, query_vector=query_vector
        )
         
        candidate_chunks = [
            point
            for point in vector_search_chunks
            if point.payload and point.payload.get("patent_id")
        ]

        qdrant_results = _dedupe_top_chunk_per_patent(candidate_chunks)
        
        # make the array of candidate patent ids, remove duplicates
        candidate_patent_ids = list(
            dict.fromkeys(point.payload["patent_id"] for point in candidate_chunks)
        )

        if parsed.metadata_filters:
            surviving_patent_ids = _run(
                "Metadata Filtering",
                self._filter_patent_ids_by_metadata,
                candidate_patent_ids,
                parsed.metadata_filters,
            )
        else:
            surviving_patent_ids = candidate_patent_ids

        # Fetch EVERY indexed chunk of each surviving candidate patent (unbounded, not just the handful the initial vector search happened to surface)
        filtered_results = _run(
            "Full Chunk Retrieval",
            self.db.get_chunks_for_patent_ids,
            surviving_patent_ids,
        )

        # Rerank EVERY one of those chunks individually; a patent's score is the MAX across its own chunks (see app/reranker.py) - checking the patent's complete text, not a similarity-biased subset.
        reranked_results = _run(
            "Reranking",
            self.reranker.rerank,
            query=parsed.semantic_query,
            results=filtered_results,
            requirements=parsed,
        )

        # keep only patents scoring >= PATENT_RELEVANCE_THRESHOLD
        reranked_results = [
            item for item in reranked_results if item[0] >= PATENT_RELEVANCE_THRESHOLD
        ]

        # Step 4b: For question queries, extract answer evidence and spans
        if parsed.is_question:
            reranked_results = _run(
                "Answer Evidence",
                self.evidence_selector.select_and_extract,
                reranked_results,
                parsed,
            )

        # aggregate by patent_id and produce one PatentSearchResult per patent
        patent_results = _run(
            "Aggregation",
            lambda: self._aggregate_by_patent(
                reranked_results, is_question=parsed.is_question, parsed_query=parsed
            )[:FINAL_TOP_K],
        )

        return (
            parsed,
            qdrant_results,
            filtered_results,
            reranked_results,
            patent_results,
        )

    def search(self, query: str) -> list[PatentSearchResult]:
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
        ParsedQuery,
        list,
        list,
        list[tuple[float, object]],
        list[PatentSearchResult],
    ]:
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

        # chunks already carry a placeholder .score of 1.0 (see
        # QdrantDB.get_chunks_for_patent_ids - "confirmed metadata
        # match", not a relevance value); rescaled to 10.0 here so it
        # reads consistently alongside the reranker's 0-10 relevance
        # scale rather than looking like a near-zero score.
        reranked_results = [(10.0, chunk) for chunk in chunks]

        patent_results = _run(
            "Aggregation",
            lambda: self._aggregate_by_patent(
                reranked_results, is_question=False, parsed_query=parsed
            ),
        )

        # qdrant_results (2nd position) is the patent-level candidate view
        # the UI displays as "Qdrant Vector Search Candidates" - dedupe to
        # one chunk per patent_id here too, same as the vector-search path
        # (_dedupe_top_chunk_per_patent in search_detailed), so the count
        # shown reflects distinct matching PATENTS, not every one of their
        # chunks. `chunks` (3rd position, filtered_results) stays the full,
        # unbounded set - that's what aggregation above actually used.
        qdrant_results = _dedupe_top_chunk_per_patent(chunks)

        return parsed, qdrant_results, chunks, reranked_results, patent_results

    # ==============================================================
    # Metadata filtering (post-vector-search, pre-chunk-retrieval)
    # ==============================================================

    def _filter_patent_ids_by_metadata(
        self,
        patent_ids: list[str],
        metadata_filters: list,
    ) -> list[str]:
        """
        Keep only candidate patent_ids whose stored metadata satisfies
        every filter.

        Fetches metadata for *patent_ids* in ONE batch call (the
        existing get_patents_metadata - no per-patent requests) and
        evaluates FilterEngine.matches() per patent. A patent with no
        stored metadata can't satisfy any filter and is excluded.
        Runs BEFORE any chunk text is fetched, so a patent's full chunk
        set is only ever pulled for patents that already pass this.
        """

        patent_metadata = self.db.get_patents_metadata(patent_ids)

        return [
            patent_id
            for patent_id in patent_ids
            if FilterEngine.matches(patent_metadata.get(patent_id, {}), metadata_filters)
        ]

    # ==============================================================
    # Patent aggregation
    # ==============================================================

    def _aggregate_by_patent(
        self,
        reranked: list[tuple[float, object]],
        is_question: bool = False,
        parsed_query: ParsedQuery | None = None,
    ) -> list[PatentSearchResult]:
        """
        Group reranked chunks by patent_id and produce one
        PatentSearchResult per patent.

        Every chunk of a patent shares that patent's relevance score -
        the MAX across all of that patent's own chunks (see
        app/reranker.py) - so `best_chunk` is chosen by each chunk's
        OWN individual score instead (RankedChunk.chunk_relevance_score,
        stashed by the reranker on payload["_chunk_relevance"]): that's
        exactly the chunk that earned the patent's max score, decided
        by the same relevance model, not a separate heuristic.
        """

        if not reranked:
            return []

        # ---- Group chunks by patent_id ----
        patent_chunks: dict[str, list[RankedChunk]] = defaultdict(list)

        for score, result in reranked:
            payload = getattr(result, "payload", {}) or {}
            patent_id = payload.get("patent_id", "")

            ranked_chunk = RankedChunk(
                chunk_id=payload.get("chunk_id", 0),
                section=payload.get("section", ""),
                text=payload.get("text", ""),
                score=score,
                chunk_relevance_score=payload.get("_chunk_relevance", 0.0) or 0.0,
                token_count=payload.get("token_count", 0),
                word_count=payload.get("word_count", 0),
                section_chunk_index=payload.get("section_chunk_index", 0),
                document_chunk_index=payload.get("document_chunk_index", 0),
                total_chunks=payload.get("total_chunks", 0),
                answer=payload.get("answer"),
                answer_evidence=payload.get("answer_evidence") or [],
                answer_span=payload.get("answer_span"),
                answer_score=payload.get("answer_score"),
                highlighted_text=payload.get("highlighted_text"),
                answer_debug=payload.get("answer_debug"),
            )

            patent_chunks[patent_id].append(ranked_chunk)

        # ---- Fetch metadata once per unique patent ----
        patent_metadata = self.db.get_patents_metadata(list(patent_chunks.keys()))

        # ---- Build PatentSearchResult per patent ----
        patent_results: list[PatentSearchResult] = []

        for patent_id, chunks in patent_chunks.items():
            # Pick the displayed chunk by its OWN individual relevance
            # score - the patent-level score is identical across a
            # patent's own chunks (it's the max of them all), so it
            # can't be the tiebreaker here; chunk_relevance_score is
            # exactly the score that produced that max for whichever
            # chunk earned it.
            chunks.sort(key=lambda c: c.chunk_relevance_score, reverse=True)

            best = chunks[0]

            answer_chunk = best
            answer_text = None
            answer_evidence = []
            answer_span = None
            answer_score = None
            highlighted_text = None
            answer_debug = None

            if is_question:
                answered_chunks = [c for c in chunks if (c.answer_score or 0.0) > 0.0]
                if answered_chunks:
                    answered_chunks.sort(
                        key=lambda c: ((c.answer_score or 0.0), c.chunk_relevance_score),
                        reverse=True,
                    )
                    answer_chunk = answered_chunks[0]

                answer_text = answer_chunk.answer
                answer_evidence = answer_chunk.answer_evidence
                answer_span = answer_chunk.answer_span
                answer_score = answer_chunk.answer_score
                highlighted_text = answer_chunk.highlighted_text
                answer_debug = answer_chunk.answer_debug

                # For comparison-style questions, merge top two direct answers when available.
                query_types = [
                    qt.lower()
                    for qt in (parsed_query.query_type if parsed_query else [])
                ]
                is_comparative = any(
                    qt in ("comparative", "tradeoff") for qt in query_types
                )
                if is_comparative and len(answered_chunks) >= 2:
                    first = answered_chunks[0]
                    second = answered_chunks[1]
                    if first.answer and second.answer and first.answer != second.answer:
                        answer_text = f"{first.answer} | {second.answer}"
                        answer_evidence = [
                            *first.answer_evidence,
                            *second.answer_evidence,
                        ]
                        answer_score = max(
                            first.answer_score or 0.0, second.answer_score or 0.0
                        )
                        highlighted_text = (
                            first.highlighted_text or second.highlighted_text
                        )
                        answer_debug = {
                            "mode": "multi_chunk_comparison",
                            "selected_chunks": [first.chunk_id, second.chunk_id],
                            "combined_answer": answer_text,
                            "primary": first.answer_debug,
                            "secondary": second.answer_debug,
                        }

            patent_results.append(
                PatentSearchResult(
                    patent_id=patent_id,
                    score=best.score,
                    best_chunk=best,
                    matching_chunks=chunks,
                    metadata=patent_metadata.get(patent_id, {}),
                    answer=answer_text if is_question else None,
                    answer_evidence=answer_evidence if is_question else [],
                    answer_span=answer_span if is_question else None,
                    answer_score=answer_score if is_question else None,
                    highlighted_text=highlighted_text if is_question else None,
                    answer_debug=answer_debug if is_question else None,
                )
            )

        # ---- Sort patents: question queries put a genuinely answered
        # patent first (that's the whole point of asking a question),
        # then relevance score; topic queries sort by relevance score
        # alone. ----
        def _patent_rank_key(p: PatentSearchResult) -> tuple[int, float]:
            has_answer = 1 if (is_question and p.has_answer) else 0
            return (has_answer, p.score)

        patent_results.sort(key=_patent_rank_key, reverse=True)

        return patent_results

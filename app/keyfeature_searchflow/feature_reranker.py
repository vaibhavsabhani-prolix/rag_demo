"""
Key Feature Reranker.

Reranks candidate chunks retrieved for each key feature against that feature's
text using the BGE cross-encoder reranker, and selects the top-K final chunks.
"""

import logging
import re
from collections.abc import Callable

from app.config import CHUNKS_COLLECTION_NAME
from app.highlighting import (
    extract_term_patterns,
    find_term_spans,
    sentences_to_score,
    split_sentences,
)
from app.keyfeature_searchflow.models import (
    FeatureChunkResult,
    FeaturePatentResult,
    FeatureSearchResult,
    StepPatentInfo,
)
from app.models.reranking import ChunkHighlight, HighlightSentence, HighlightTerm
from app.qdrant_db import QdrantDB
from app.reranking.reranker import BGEReranker

logger = logging.getLogger(__name__)

# Payload fields needed to rebuild a chunk for reranking/display.
_PATENT_CHUNK_PAYLOAD_FIELDS = [
    "patent_id",
    "chunk_id",
    "text",
    "section",
    "document_chunk_index",
    "token_count",
    "title",
]

# Debug/explain entries (vector_retrieval_patents, rerank_filtered_patents,
# topk_pruned_patents) are for non-winning patents and only need a preview,
# not every chunk of the patent. Capping this keeps the response small even
# when a patent's full chunk set is fetched for reranking.
_MAX_DEBUG_CHUNKS_PER_PATENT = 3


class KeyFeatureReranker:
    def __init__(self, reranker: BGEReranker | None = None, db: QdrantDB | None = None):
        self.reranker = reranker or BGEReranker()
        self.db = db or QdrantDB()

    def _format_chunk(
        self, text: str, section: str | None, doc_budget: int = 3800
    ) -> str:
        """
        Format chunk with section prefix and ensure it fits within context budget.
        """
        prefix = f"Section: {section}\n\n" if section else ""
        combined = f"{prefix}{text}" if prefix else text
        # Fast safeguard: BGE max context is ~4100 tokens.
        # Chunks in patent_chunks_512 are <= 512 tokens.
        # If combined text is unusually large (>12000 chars ~= 3000 tokens), safely slice.
        if len(combined) > 12000:
            return combined[:12000]
        return combined

    def rerank_feature_results(
        self,
        results: list[FeatureSearchResult],
        top_k: int = 20,
        rerank_score_threshold: float = 0.6,
        chunks_collection: str = CHUNKS_COLLECTION_NAME,
        on_feature_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[FeatureSearchResult]:
        """
        Rerank retrieved candidate chunks for each feature against that feature's text,
        then select and retain the top_k unique patents (along with all matching chunks for each).

        Reranking a feature can be slow now that it pulls each matched patent's full
        chunk set from Qdrant, so `on_feature_progress(done, total, feature_text)` is
        invoked after every feature finishes - this gives callers a steady stream of
        progress events (and doubles as a keep-alive on long-running SSE connections).
        """
        if not results:
            return []

        total_features = len(results)
        for idx, feature_result in enumerate(results, start=1):
            try:
                self._rerank_one_feature(
                    feature_result=feature_result,
                    top_k=top_k,
                    rerank_score_threshold=rerank_score_threshold,
                    chunks_collection=chunks_collection,
                )
            finally:
                if on_feature_progress:
                    on_feature_progress(idx, total_features, feature_result.feature)

        return results

    def _rerank_one_feature(
        self,
        feature_result: FeatureSearchResult,
        top_k: int,
        rerank_score_threshold: float,
        chunks_collection: str,
    ) -> None:
        chunks = feature_result.chunks
        if not chunks:
            return
        feat_id = feature_result.feature_id
        for ch in chunks:
            ch.feature_id = feat_id

        # Group all chunks from initial vector retrieval by patent_id
        vector_patent_chunks: dict[str, list[FeatureChunkResult]] = {}
        for ch in chunks:
            if ch.patent_id:
                vector_patent_chunks.setdefault(ch.patent_id, []).append(ch)

        vector_retrieval_patents: list[StepPatentInfo] = []
        for pid, p_chunks in vector_patent_chunks.items():
            best_retrieval_sc = max(
                (c.retrieval_score if c.retrieval_score is not None else c.score for c in p_chunks),
                default=0.0,
            )
            vector_retrieval_patents.append(
                StepPatentInfo(
                    patent_id=pid,
                    title=p_chunks[0].title,
                    retrieval_score=round(best_retrieval_sc, 6),
                    chunks_count=len(p_chunks),
                    stage="retrieval",
                    passed=True,
                    chunks=p_chunks[:_MAX_DEBUG_CHUNKS_PER_PATENT],
                )
            )
        vector_retrieval_patents.sort(key=lambda p: p.retrieval_score or 0.0, reverse=True)
        for idx, p in enumerate(vector_retrieval_patents, start=1):
            p.rank = idx

        query = (feature_result.feature or "").strip()
        if not query:
            # If feature query is empty, group by unique patent by initial retrieval score
            for p_chunks in vector_patent_chunks.values():
                p_chunks.sort(key=lambda c: c.score, reverse=True)

            ranked_patent_items = sorted(
                vector_patent_chunks.items(),
                key=lambda item: item[1][0].score,
                reverse=True,
            )

            top_patent_items = ranked_patent_items[:top_k]
            pruned_patent_items = ranked_patent_items[top_k:]

            top_patents: list[FeaturePatentResult] = []
            retained_chunks: list[FeatureChunkResult] = []
            for pid, p_chunks in top_patent_items:
                best_chunk = p_chunks[0]
                pat_res = FeaturePatentResult(
                    patent_id=pid,
                    title=best_chunk.title,
                    score=best_chunk.score,
                    best_reranker_score=None,
                    best_retrieval_score=best_chunk.score,
                    metadata=dict(best_chunk.metadata) if best_chunk.metadata else {},
                    chunks=p_chunks,
                )
                top_patents.append(pat_res)
                retained_chunks.extend(p_chunks)

            topk_pruned_patents: list[StepPatentInfo] = []
            for rank_offset, (pat_id, p_chunks) in enumerate(pruned_patent_items, start=top_k + 1):
                best_c = p_chunks[0]
                topk_pruned_patents.append(
                    StepPatentInfo(
                        patent_id=pat_id,
                        title=best_c.title,
                        retrieval_score=round(best_c.score, 6),
                        reranker_score=None,
                        chunks_count=len(p_chunks),
                        rank=rank_offset,
                        stage="topk_pruned",
                        filter_reason=f"Rank #{rank_offset} exceeds Top-{top_k} limit",
                        passed=False,
                        chunks=p_chunks[:_MAX_DEBUG_CHUNKS_PER_PATENT],
                    )
                )

            feature_result.patents = top_patents
            feature_result.chunks = retained_chunks
            feature_result.vector_retrieval_patents = vector_retrieval_patents
            feature_result.rerank_filtered_patents = []
            feature_result.topk_pruned_patents = topk_pruned_patents
            return

        # Instead of reranking only the chunks that matched this feature's
        # vector search, pull every stored chunk belonging to each matched
        # patent and send the patent's full chunk set to reranking.
        matched_patent_ids = list(vector_patent_chunks.keys())
        full_chunks_by_patent = self.db.get_chunks_by_patent_ids(
            patent_ids=matched_patent_ids,
            chunks_collection=chunks_collection,
            payload_fields=_PATENT_CHUNK_PAYLOAD_FIELDS,
        )

        rerank_patent_chunks: dict[str, list[FeatureChunkResult]] = {}
        chunks: list[FeatureChunkResult] = []
        for pid in matched_patent_ids:
            matched_chunks = vector_patent_chunks[pid]
            seed = matched_chunks[0]
            raw_chunks = full_chunks_by_patent.get(pid) or []
            if not raw_chunks:
                # Full fetch failed/empty - fall back to the matched chunks only.
                patent_chunks = matched_chunks
            else:
                patent_chunks = [
                    FeatureChunkResult(
                        chunk_id=int(payload.get("chunk_id", 0)),
                        patent_id=pid,
                        feature_id=feat_id,
                        score=seed.score,
                        retrieval_score=seed.retrieval_score,
                        reranker_score=None,
                        text=payload.get("text", ""),
                        section=payload.get("section"),
                        title=payload.get("title") or seed.title,
                        document_chunk_index=payload.get("document_chunk_index"),
                        token_count=payload.get("token_count"),
                        metadata={},
                    )
                    for payload in raw_chunks
                ]
            rerank_patent_chunks[pid] = patent_chunks
            chunks.extend(patent_chunks)
            logger.info(
                "Feature %d: patent %s matched %d chunk(s) in vector search -> "
                "sending all %d chunk(s) of this patent to reranking",
                feat_id,
                pid,
                len(matched_chunks),
                len(patent_chunks),
            )

        formatted_docs: list[str] = [
            self._format_chunk(ch.text or "", ch.section) for ch in chunks
        ]

        logger.info(
            "Feature %d ('%s'): reranking %d chunk(s) across %d matched patent(s)",
            feature_result.feature_id,
            query[:60],
            len(chunks),
            len(rerank_patent_chunks),
        )

        # Score documents using BGE cross-encoder in concurrent batches
        scores, _ = self.reranker.score_documents(
            query=query,
            documents=formatted_docs,
        )

        has_valid_scores = any(s != 0.0 for s in scores)

        # Attach scores to chunks
        for ch, score in zip(chunks, scores):
            r_score = round(score, 6)
            ch.reranker_score = r_score
            if has_valid_scores:
                ch.score = r_score

        # Identify patents that fail the rerank threshold vs those that qualify
        rerank_filtered_patents: list[StepPatentInfo] = []
        qualifying_patent_chunks_map: dict[str, list[FeatureChunkResult]] = {}

        for pid, p_chunks in rerank_patent_chunks.items():
            p_chunks.sort(
                key=lambda c: (
                    c.reranker_score if c.reranker_score is not None else -1.0,
                    c.retrieval_score if c.retrieval_score is not None else (c.score or 0.0),
                ),
                reverse=True,
            )
            best_r_score = max(
                (c.reranker_score for c in p_chunks if c.reranker_score is not None),
                default=None,
            )
            best_retrieval_sc = max(
                (c.retrieval_score for c in p_chunks if c.retrieval_score is not None),
                default=0.0,
            )
            # Check if patent fails threshold
            if (
                rerank_score_threshold > 0
                and has_valid_scores
                and (best_r_score is None or best_r_score < rerank_score_threshold)
            ):
                rerank_filtered_patents.append(
                    StepPatentInfo(
                        patent_id=pid,
                        title=p_chunks[0].title,
                        retrieval_score=round(best_retrieval_sc, 6),
                        reranker_score=round(best_r_score, 6) if best_r_score is not None else None,
                        chunks_count=len(p_chunks),
                        stage="rerank_filtered",
                        filter_reason=f"Below rerank threshold ({best_r_score:.3f} < {rerank_score_threshold:.2f})"
                        if best_r_score is not None
                        else "No rerank score",
                        passed=False,
                        chunks=p_chunks[:_MAX_DEBUG_CHUNKS_PER_PATENT],
                    )
                )
            else:
                # Chunks that qualify for this patent
                valid_chunks = [
                    c for c in p_chunks
                    if not rerank_score_threshold or not has_valid_scores or (c.reranker_score is not None and c.reranker_score >= rerank_score_threshold)
                ]
                if valid_chunks:
                    valid_chunks.sort(
                        key=lambda c: (
                            c.reranker_score if c.reranker_score is not None else 0.0,
                            c.retrieval_score if c.retrieval_score is not None else 0.0,
                        ),
                        reverse=True,
                    )
                    qualifying_patent_chunks_map[pid] = valid_chunks

        # Rank qualifying unique patents by best chunk's reranker score
        ranked_patent_items = sorted(
            qualifying_patent_chunks_map.items(),
            key=lambda item: (
                item[1][0].reranker_score if item[1][0].reranker_score is not None else 0.0,
                item[1][0].retrieval_score if item[1][0].retrieval_score is not None else 0.0,
            ),
            reverse=True,
        )

        # Select top_k unique patents and separate pruned ones
        top_patent_items = ranked_patent_items[:top_k]
        pruned_patent_items = ranked_patent_items[top_k:]

        topk_pruned_patents: list[StepPatentInfo] = []
        for rank_offset, (pat_id, p_chunks) in enumerate(pruned_patent_items, start=top_k + 1):
            best_c = p_chunks[0]
            topk_pruned_patents.append(
                StepPatentInfo(
                    patent_id=pat_id,
                    title=best_c.title,
                    retrieval_score=round(best_c.retrieval_score, 6) if best_c.retrieval_score is not None else None,
                    reranker_score=round(best_c.reranker_score, 6) if best_c.reranker_score is not None else None,
                    chunks_count=len(p_chunks),
                    rank=rank_offset,
                    stage="topk_pruned",
                    filter_reason=f"Rank #{rank_offset} exceeds Top-{top_k} limit",
                    passed=False,
                    chunks=p_chunks[:_MAX_DEBUG_CHUNKS_PER_PATENT],
                )
            )

        # Sort filtered patents by score
        rerank_filtered_patents.sort(
            key=lambda p: (p.reranker_score if p.reranker_score is not None else -1.0, p.retrieval_score or 0.0),
            reverse=True,
        )

        # Build FeaturePatentResult objects with ALL matching chunks for each winning patent
        top_patents: list[FeaturePatentResult] = []
        retained_chunks: list[FeatureChunkResult] = []

        for pat_id, p_chunks in top_patent_items:
            best_chunk = p_chunks[0]
            best_retrieval = max(
                (c.retrieval_score for c in p_chunks if c.retrieval_score is not None),
                default=None,
            )
            pat_result = FeaturePatentResult(
                patent_id=pat_id,
                title=best_chunk.title,
                score=best_chunk.score,
                best_reranker_score=best_chunk.reranker_score,
                best_retrieval_score=best_retrieval,
                metadata=dict(best_chunk.metadata) if best_chunk.metadata else {},
                chunks=p_chunks,
            )
            top_patents.append(pat_result)
            retained_chunks.extend(p_chunks)

        feature_result.patents = top_patents
        feature_result.chunks = retained_chunks
        feature_result.vector_retrieval_patents = vector_retrieval_patents
        feature_result.rerank_filtered_patents = rerank_filtered_patents
        feature_result.topk_pruned_patents = topk_pruned_patents

        # Compute sentence-level highlights and query-term matches on all retained chunks
        self._compute_highlights_for_chunks(query=query, chunks=retained_chunks)

        best_score = top_patents[0].score if top_patents else 0.0
        logger.info(
            "Feature %d: retained top %d unique patents (%d total matching chunks) out of %d candidate chunks across %d patents (best score: %.4f, threshold: %.4f)",
            feature_result.feature_id,
            len(top_patents),
            len(retained_chunks),
            len(chunks),
            len(vector_patent_chunks),
            best_score,
            rerank_score_threshold,
        )

    def _compute_highlights_for_chunks(
        self,
        query: str,
        chunks: list[FeatureChunkResult],
    ) -> None:
        """
        Compute sentence-level BGE scores and query term matches for winning chunks,
        ensuring the top matching sentence is always highlighted.
        """
        if not chunks or not query:
            return

        # 1. Split each chunk into candidate sentences
        chunk_spans: dict[int, list[tuple[int, int]]] = {}
        all_sentence_docs: list[str] = []

        for idx, ch in enumerate(chunks):
            text = ch.text or ""
            spans = sentences_to_score(text)
            if not spans:
                spans = split_sentences(text)
            chunk_spans[idx] = spans
            for s, e in spans:
                all_sentence_docs.append(text[s:e])

        if not all_sentence_docs:
            return

        # 2. Score distinct sentences against the feature query
        unique_sentences = list(dict.fromkeys(all_sentence_docs))
        score_by_sentence: dict[str, float] = {}
        try:
            sent_scores, _ = self.reranker.score_documents(
                query=query,
                documents=unique_sentences,
            )
            score_by_sentence = dict(zip(unique_sentences, sent_scores))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not score sentences for highlighting: %s", exc)

        # 3. Build term match regex pattern
        term_pattern: re.Pattern[str] | None = extract_term_patterns(query)

        # 4. Attach ChunkHighlight to each chunk (ONLY the single top-1 sentence per chunk)
        for idx, ch in enumerate(chunks):
            text = ch.text or ""
            spans = chunk_spans.get(idx, [])
            scores = [score_by_sentence.get(text[s:e], 0.0) for s, e in spans]

            sentences: list[HighlightSentence] = []
            terms: list[HighlightTerm] = []

            # Identify the single #1 most matching sentence in this chunk
            if scores:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                best_span = spans[best_idx]
                best_score = scores[best_idx]

                sentences.append(
                    HighlightSentence(
                        start=best_span[0],
                        end=best_span[1],
                        score=round(best_score, 4),
                        strong=True,
                    )
                )

            # Underline all matched query words across the entire chunk
            if term_pattern:
                terms = [
                    HighlightTerm(start=s, end=e)
                    for s, e in find_term_spans(text, term_pattern)
                ]

            ch.highlight = ChunkHighlight(sentences=sentences, terms=terms)

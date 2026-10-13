"""
Key Feature Reranker.

Reranks candidate chunks retrieved for each key feature against that feature's
text using the BGE cross-encoder reranker, and selects the top-K final chunks.
"""

import logging
import re

from app.config import KEYFEATURE_RERANK_TOP_K
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
)
from app.models.reranking import ChunkHighlight, HighlightSentence, HighlightTerm
from app.reranking.reranker import BGEReranker

logger = logging.getLogger(__name__)


class KeyFeatureReranker:
    def __init__(self, reranker: BGEReranker | None = None):
        self.reranker = reranker or BGEReranker()

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
        top_k: int = KEYFEATURE_RERANK_TOP_K,
    ) -> list[FeatureSearchResult]:
        """
        Rerank retrieved candidate chunks for each feature against that feature's text,
        then select and retain the top_k unique patents (along with all matching chunks for each).
        """
        if not results:
            return []

        for feature_result in results:
            chunks = feature_result.chunks
            if not chunks:
                continue

            query = (feature_result.feature or "").strip()
            if not query:
                # If feature query is empty, group by unique patent by initial retrieval score
                patent_chunks_map: dict[str, list[FeatureChunkResult]] = {}
                for ch in chunks:
                    if ch.patent_id:
                        patent_chunks_map.setdefault(ch.patent_id, []).append(ch)

                for p_chunks in patent_chunks_map.values():
                    p_chunks.sort(key=lambda c: c.score, reverse=True)

                ranked_patent_items = sorted(
                    patent_chunks_map.items(),
                    key=lambda item: item[1][0].score,
                    reverse=True,
                )[:top_k]

                top_patents: list[FeaturePatentResult] = []
                retained_chunks: list[FeatureChunkResult] = []
                for pid, p_chunks in ranked_patent_items:
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

                feature_result.patents = top_patents
                feature_result.chunks = retained_chunks
                continue

            formatted_docs: list[str] = [
                self._format_chunk(ch.text or "", ch.section) for ch in chunks
            ]

            logger.info(
                "Reranking %d chunks for feature %d: '%s'",
                len(chunks),
                feature_result.feature_id,
                query[:60],
            )

            # Score documents using BGE cross-encoder in concurrent batches
            scores, _ = self.reranker.score_documents(
                query=query,
                documents=formatted_docs,
            )

            has_valid_scores = any(s != 0.0 for s in scores)

            # Attach scores to chunks
            for ch, score in zip(chunks, scores):
                ch.retrieval_score = ch.score  # Preserve vector similarity score
                r_score = round(score, 6)
                ch.reranker_score = r_score
                if has_valid_scores:
                    ch.score = r_score

            # Group candidate chunks by unique patent_id
            patent_chunks_map: dict[str, list[FeatureChunkResult]] = {}
            for ch in chunks:
                if not ch.patent_id:
                    continue
                patent_chunks_map.setdefault(ch.patent_id, []).append(ch)

            # Sort chunks within each patent by reranker score (tiebreaker retrieval score)
            for p_chunks in patent_chunks_map.values():
                p_chunks.sort(
                    key=lambda c: (
                        c.reranker_score if c.reranker_score is not None else 0.0,
                        c.retrieval_score if c.retrieval_score is not None else 0.0,
                    ),
                    reverse=True,
                )

            # Rank unique patents by best chunk's reranker score (tiebreaker retrieval score)
            ranked_patent_items = sorted(
                patent_chunks_map.items(),
                key=lambda item: (
                    item[1][0].reranker_score if item[1][0].reranker_score is not None else 0.0,
                    item[1][0].retrieval_score if item[1][0].retrieval_score is not None else 0.0,
                ),
                reverse=True,
            )

            # Select top_k unique patents
            top_patent_items = ranked_patent_items[:top_k]

            # Build FeaturePatentResult objects with ALL matching chunks for each patent
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

            # Compute sentence-level highlights and query-term matches on all retained chunks
            self._compute_highlights_for_chunks(query=query, chunks=retained_chunks)

            best_score = top_patents[0].score if top_patents else 0.0
            logger.info(
                "Feature %d: retained top %d unique patents (%d total matching chunks) out of %d candidate chunks across %d patents (best score: %.4f)",
                feature_result.feature_id,
                len(top_patents),
                len(retained_chunks),
                len(chunks),
                len(patent_chunks_map),
                best_score,
            )

        return results

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

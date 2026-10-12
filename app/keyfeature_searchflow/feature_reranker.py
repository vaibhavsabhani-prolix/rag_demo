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
from app.keyfeature_searchflow.models import FeatureChunkResult, FeatureSearchResult
from app.models.reranking import ChunkHighlight, HighlightSentence, HighlightTerm
from app.reranking.reranker import BGEReranker

logger = logging.getLogger(__name__)


class KeyFeatureReranker:
    def __init__(self, reranker: BGEReranker | None = None):
        self.reranker = reranker or BGEReranker()

    def _format_chunk(self, text: str, section: str | None, doc_budget: int = 3800) -> str:
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
        then select and retain only the top_k final chunks.
        """
        if not results:
            return []

        for feature_result in results:
            chunks = feature_result.chunks
            if not chunks:
                continue

            query = (feature_result.feature or "").strip()
            if not query:
                # If feature query is empty, keep top_k by initial retrieval score
                feature_result.chunks = chunks[:top_k]
                continue

            formatted_docs: list[str] = [
                self._format_chunk(ch.text or "", ch.section)
                for ch in chunks
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

            # Sort descending by reranker_score (with retrieval_score as tiebreaker)
            chunks.sort(
                key=lambda c: (
                    c.reranker_score if c.reranker_score is not None else 0.0,
                    c.retrieval_score if c.retrieval_score is not None else 0.0,
                ),
                reverse=True,
            )

            # Keep top_k final chunks
            top_chunks = chunks[:top_k]
            feature_result.chunks = top_chunks

            # Compute sentence-level highlights and query-term matches on retained chunks
            self._compute_highlights_for_chunks(query=query, chunks=top_chunks)

            best_score = feature_result.chunks[0].score if feature_result.chunks else 0.0
            logger.info(
                "Feature %d: retained top %d chunks out of %d candidates (best score: %.4f)",
                feature_result.feature_id,
                len(feature_result.chunks),
                len(chunks),
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

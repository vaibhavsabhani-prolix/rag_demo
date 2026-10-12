"""
Key Feature Reranker.

Reranks candidate chunks retrieved for each key feature against that feature's
text using the BGE cross-encoder reranker, and selects the top-K final chunks.
"""

import logging

from app.config import KEYFEATURE_RERANK_TOP_K
from app.keyfeature_searchflow.models import FeatureSearchResult
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
            feature_result.chunks = chunks[:top_k]

            best_score = feature_result.chunks[0].score if feature_result.chunks else 0.0
            logger.info(
                "Feature %d: retained top %d chunks out of %d candidates (best score: %.4f)",
                feature_result.feature_id,
                len(feature_result.chunks),
                len(chunks),
                best_score,
            )

        return results

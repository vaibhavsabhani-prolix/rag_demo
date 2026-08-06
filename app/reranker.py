"""
Qwen Reranker

Uses the official sentence-transformers CrossEncoder API.

Requires: sentence-transformers >= 5.4.0
          transformers >= 4.51.0

The Qwen3-Reranker is a CausalLM-based reranker that uses a LogitScore
module (yes/no token logit difference) rather than a classification head.
This module was introduced in sentence-transformers 5.4.0.
"""

from sentence_transformers import CrossEncoder

from app.config import (
    RERANKER_MODEL,
    FINAL_TOP_K,
)


class Reranker:
    """
    Rerank Qdrant search results using a cross-encoder.
    """

    def __init__(self):

        print(f"Loading reranker: {RERANKER_MODEL}")

        self.model = CrossEncoder(
            RERANKER_MODEL,
            trust_remote_code=True,
        )

        print("Reranker loaded successfully.\n")

    def rerank(
        self,
        query: str,
        results: list,
    ) -> list[tuple[float, object]]:
        """
        Rerank Qdrant search results using the cross-encoder.

        Returns a list of ``(reranker_score, qdrant_result)`` tuples,
        sorted by reranker score descending, truncated to FINAL_TOP_K.
        """

        if not results:
            return []

        pairs = [
            (query, result.payload["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            (float(score), result)
            for score, result in ranked[:FINAL_TOP_K]
        ]
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

from app.config import RERANKER_MODEL


class Reranker:
    """
    Rerank Qdrant search results using a cross-encoder.
    """

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL,
            trust_remote_code=True,
        )

    def rerank(
        self,
        query: str,
        results: list,
    ) -> list[tuple[float, object]]:
        """
        Rerank Qdrant search results using the cross-encoder.

        Returns every result as a ``(reranker_score, qdrant_result)``
        tuple, sorted by reranker score descending. Not truncated here -
        results still need to be aggregated into patents (a patent's
        score is its best chunk's score), and cutting to a fixed chunk
        count first could drop a patent whose only strong chunk missed
        that cut. Truncation to the final patent count happens after
        aggregation instead (see SemanticSearch.search_detailed).
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
            for score, result in ranked
        ]
"""
Reranker

Choose between a remote reranking server and a local sentence-transformers
CrossEncoder using a single config flag.
"""

import requests
from sentence_transformers import CrossEncoder

from app.config import (
    LOCAL_RERANKER_MODEL,
    RERANKER_REMOTE_API_KEY,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RERANKER_REQUEST_TIMEOUT,
    USE_REMOTE_RERANKER,
)


class Reranker:
    """
    Rerank Qdrant search results using either the remote server or a local model.
    """

    def __init__(self, use_remote: bool | None = None):
        self.use_remote = USE_REMOTE_RERANKER if use_remote is None else use_remote

        if self.use_remote:
            self.base_url = RERANKER_REMOTE_BASE_URL.rstrip("/")
            self.model = RERANKER_REMOTE_MODEL
            self.headers = {"Authorization": f"Bearer {RERANKER_REMOTE_API_KEY}"}
            self.cross_encoder = None
        else:
            self.cross_encoder = CrossEncoder(LOCAL_RERANKER_MODEL)
            self.base_url = None
            self.model = LOCAL_RERANKER_MODEL
            self.headers = {}

    def rerank(
        self,
        query: str,
        results: list,
    ) -> list[tuple[float, object]]:
        """
        Rerank Qdrant search results.

        Returns every result as a ``(reranker_score, qdrant_result)``
        tuple, sorted by reranker score descending.
        """

        if not results:
            return []

        texts = [result.payload["text"] for result in results]
        print(f"Reranking {len(texts)} candidate chunks for query: {query}")

        if self.use_remote:
            response = requests.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                },
                timeout=RERANKER_REQUEST_TIMEOUT,
            )
            print(f"Reranker response status code: {response.status_code}")
            response.raise_for_status()

            ranked = sorted(
                response.json()["results"],
                key=lambda item: item["relevance_score"],
                reverse=True,
            )

            return [
                (float(item["relevance_score"]), results[item["index"]])
                for item in ranked
            ]

        pairs = [(query, text) for text in texts]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        return [
            (float(scores[idx]), results[idx])
            for idx in ranked_indices
        ]

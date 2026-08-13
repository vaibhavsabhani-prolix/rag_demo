"""
Remote Reranker

Calls a remotely-hosted reranking server over HTTP instead of loading a
cross-encoder model locally. Expects a vLLM-style OpenAI-compatible
``/rerank`` endpoint:

    POST {base_url}/rerank
    {"model": ..., "query": ..., "documents": [...]}

    -> {"results": [{"index": 0, "relevance_score": 0.98}, ...]}
"""

import requests

from app.config import (
    RERANKER_REMOTE_API_KEY,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RERANKER_REQUEST_TIMEOUT,
)


class Reranker:
    """
    Rerank Qdrant search results using a remote reranking server.
    """

    def __init__(self):

        self.base_url = RERANKER_REMOTE_BASE_URL.rstrip("/")
        self.model = RERANKER_REMOTE_MODEL
        self.headers = {"Authorization": f"Bearer {RERANKER_REMOTE_API_KEY}"}

    def rerank(
        self,
        query: str,
        results: list,
    ) -> list[tuple[float, object]]:
        """
        Rerank Qdrant search results using the remote reranker.

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

        texts = [result.payload["text"] for result in results]
        print(f"Reranking {len(texts)} candidate chunks for query: {query}")
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

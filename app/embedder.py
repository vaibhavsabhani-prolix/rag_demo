"""
Embedding Model (Remote)

Calls a remote vLLM embedding server via its OpenAI-compatible
/embeddings endpoint.

Performance notes:

    - EMBED_BATCH_SIZE is the number of texts per HTTP request. With a
      remote DGX, this should be large (256+) to amortize network
      round-trip latency and keep the GPU busy.

    - EMBED_CONCURRENT_REQUESTS fires multiple HTTP requests in
      parallel via a thread pool, so the next batch is already in
      transit while the current one computes on the GPU.

    - A requests.Session reuses TCP connections across calls, avoiding
      the per-request overhead of a fresh TCP/TLS handshake.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from app.config import (
    EMBED_BATCH_SIZE,
    EMBED_CONCURRENT_REQUESTS,
    EMBEDDING_REMOTE_API_KEY,
    EMBEDDING_REMOTE_BASE_URL,
    EMBEDDING_REMOTE_MODEL,
    EMBEDDING_REQUEST_TIMEOUT,
)
from app.models.patent_chunk import PatentChunk


class Embedder:
    def __init__(self):
        self.max_workers = EMBED_CONCURRENT_REQUESTS
        self._init_remote()

    # ==============================================================
    # Initialisation helpers
    # ==============================================================

    def _init_remote(self):
        """Set up an HTTP session for the remote vLLM server."""
        self.base_url = EMBEDDING_REMOTE_BASE_URL.rstrip("/")
        self.model = EMBEDDING_REMOTE_MODEL
        self.max_workers = EMBED_CONCURRENT_REQUESTS

        # Persistent session — reuses TCP connections across requests
        # to the same host, skipping per-call handshake overhead.
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {EMBEDDING_REMOTE_API_KEY}",
                "Content-Type": "application/json",
            }
        )

        self._check_remote()

        print(f"Using remote embedding model: {self.model}")
        print(f"  Base URL: {self.base_url}")
        print(
            f"  Batch size: {EMBED_BATCH_SIZE}, concurrent requests: {self.max_workers}"
        )

    def _check_remote(self):
        """Fail before ingestion starts when the embedding service is down."""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=min(10.0, EMBEDDING_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Remote embedding service is unavailable at "
                f"{self.base_url}. Start the vLLM server or verify the host "
                "and port."
            ) from exc

    # ==============================================================
    # Remote API call
    # ==============================================================

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a batch of texts and return one vector per text,
        index-aligned with the input.
        """
        return self._encode_remote(texts)

    def _encode_remote(self, texts: list[str]) -> list[list[float]]:
        """
        Send a batch of texts to the remote /embeddings endpoint and
        return one vector per text, index-aligned with the input.
        """

        response = self.session.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=EMBEDDING_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()["data"]
        # The OpenAI API returns objects with an "index" field; sort by
        # index to guarantee alignment with the input list.
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    # ==============================================================
    # Chunk embedding (ingestion)
    # ==============================================================

    def embed(self, chunk: PatentChunk) -> PatentChunk:
        """
        Generate an embedding for a single PatentChunk.
        """

        self.embed_batch([chunk])
        return chunk

    def embed_batch(
        self,
        chunks: list[PatentChunk],
        on_progress=None,
    ) -> list[PatentChunk]:
        """
        Generate embeddings for many PatentChunks.

        Splits the input into EMBED_BATCH_SIZE windows and fires up to
        EMBED_CONCURRENT_REQUESTS of them in parallel via a thread
        pool. Each window is an independent HTTP call to the remote
        server; the server's GPU processes them as capacity allows.

        *on_progress*, if given, is called with the number of chunks
        completed as each window finishes. Progress reports arrive as
        futures complete (possibly out of submission order), but every
        vector is assigned to the correct chunk regardless.

        Vectors are assigned back onto the chunks in place; the same
        list is returned for convenience.
        """

        if not chunks:
            return chunks

        windows = [
            chunks[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(chunks), EMBED_BATCH_SIZE)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._encode_into, window): window for window in windows
            }

            for future in as_completed(futures):
                future.result()  # propagate exceptions
                if on_progress is not None:
                    on_progress(len(futures[future]))

        return chunks

    def _encode_into(self, chunks: list[PatentChunk]) -> None:
        """Encode *chunks* and assign each vector back."""

        texts = [chunk.text for chunk in chunks]
        vectors = self._request_embeddings(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector

    # ==============================================================
    # Query embedding (search time)
    # ==============================================================

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a search query.
        """

        vectors = self._request_embeddings([query])
        return vectors[0]

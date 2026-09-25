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

    def close(self) -> None:
        """Release the underlying HTTP session's connections."""
        self.session.close()

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

        # The default adapter pools only 10 connections per host; with
        # more concurrent workers than that, the extras are opened and
        # thrown away on every request instead of being reused.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=max(10, self.max_workers),
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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

    def _request_embeddings(
        self, texts: list[str], batch_label: str = "1/1"
    ) -> list[list[float]]:
        """
        Encode a batch of texts and return one vector per text,
        index-aligned with the input.
        """
        return self._encode_remote(texts, batch_label)

    def _encode_remote(
        self, texts: list[str], batch_label: str = "1/1"
    ) -> list[list[float]]:
        """
        Send a batch of texts to the remote /embeddings endpoint and
        return one vector per text, index-aligned with the input.
        """

        print(
            f"[Embedder] Request {batch_label} -> {self.base_url}/embeddings"
            f" ({len(texts)} texts)"
        )

        response = self.session.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=EMBEDDING_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        print(f"[Embedder] Request {batch_label} <- HTTP {response.status_code}")

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
        label: str | None = None,
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

        *label*, if given, identifies this call in the request log
        (e.g. a caller-assigned window number) instead of the default
        "i/N" split label - useful when the caller already guarantees
        each call is a single window, so "i/N" would always read "1/1"
        and give no sense of progress across calls.

        Vectors are assigned back onto the chunks in place; the same
        list is returned for convenience.
        """

        if not chunks:
            return chunks

        windows = [
            chunks[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(chunks), EMBED_BATCH_SIZE)
        ]

        def _label(i: int) -> str:
            if label is None:
                return f"{i + 1}/{len(windows)}"
            if len(windows) == 1:
                return label
            return f"{label}.{i + 1}/{len(windows)}"

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._encode_into, window, _label(i)): window
                for i, window in enumerate(windows)
            }

            for future in as_completed(futures):
                future.result()  # propagate exceptions
                if on_progress is not None:
                    on_progress(len(futures[future]))

        return chunks

    def _encode_into(self, chunks: list[PatentChunk], batch_label: str = "1/1") -> None:
        """Encode *chunks* and assign each vector back."""

        texts = [chunk.text for chunk in chunks]
        vectors = self._request_embeddings(texts, batch_label)

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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of query strings / retrieval views
        in a single batch request to avoid repeated HTTP calls.
        """
        if not texts:
            return []
        return self._request_embeddings(texts)

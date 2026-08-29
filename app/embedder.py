"""
Embedding Model
"""

import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL, MAX_CHUNK_TOKENS
from app.models.patent_chunk import PatentChunk


class Embedder:
    def __init__(self):

        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )

        # Chunks are token-bounded by the chunker (MAX_CHUNK_TOKENS), so
        # the model's default context window (32k for Qwen3-Embedding) is
        # far larger than anything it will ever be handed here. Capping it
        # costs no content but keeps the tokenizer from allocating for a
        # window this pipeline never uses.
        self.model.max_seq_length = MAX_CHUNK_TOKENS

        # Inference only - no autograd graph needed.
        self.model.eval()

        print("Model loaded successfully.\n")

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
        Generate embeddings for many PatentChunks in one forward pass.

        Encoding chunks together lets the model run one batched matrix
        multiply instead of one per chunk, and lets SentenceTransformers
        sort by length internally so each batch pads to its own longest
        member rather than to the longest chunk overall. This is the
        ingestion hot path - embedding dominates total ingest time.

        *on_progress*, if given, is called with the number of chunks
        completed after each batch. A single patent can produce
        thousands of chunks and take hours to embed, so the caller needs
        a way to report movement inside that. Supplying it encodes in
        explicit EMBED_BATCH_SIZE windows instead of handing the whole
        list to SentenceTransformers at once, which gives up its global
        length-sorting - a non-issue here, where the chunker already
        caps every chunk at MAX_CHUNK_TOKENS and most land near it.

        Vectors are assigned back onto the chunks in place; the same
        list is returned for convenience.
        """

        if not chunks:
            return chunks

        if on_progress is None:
            self._encode_into(chunks)
            return chunks

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            window = chunks[start:start + EMBED_BATCH_SIZE]

            self._encode_into(window)

            on_progress(len(window))

        return chunks

    def _encode_into(self, chunks: list[PatentChunk]) -> None:
        """Encode *chunks* and assign each vector back onto its chunk."""

        with torch.inference_mode():
            vectors = self.model.encode(
                [chunk.text for chunk in chunks],
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a search query.
        """

        with torch.inference_mode():
            vector = self.model.encode(
                query,
                normalize_embeddings=True,
            )

        return vector.tolist()

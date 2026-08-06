"""
Embedding Model
"""

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.models.patent_chunk import PatentChunk


class Embedder:

    def __init__(self):

        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )

        print("Model loaded successfully.\n")

    def embed(self, chunk: PatentChunk) -> PatentChunk:
        """
        Generate an embedding for a PatentChunk.
        """

        vector = self.model.encode(
            chunk.text,
            normalize_embeddings=True,
        )

        chunk.vector = vector.tolist()

        return chunk

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a search query.
        """

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return vector.tolist()
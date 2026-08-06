"""
Qdrant Database Layer
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import (
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_SIZE,
    VECTOR_TOP_K,
)

from app.models.patent_chunk import PatentChunk


class QdrantDB:

    def __init__(self):

        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

    def create_collection(self):
        """
        Create the collection if it doesn't already exist.
        """

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME in names:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(f"Collection '{COLLECTION_NAME}' created.")

    def reset_collection(self):
        """
        Delete and recreate the collection.
        """

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME in names:

            print(f"Deleting collection '{COLLECTION_NAME}'...")

            self.client.delete_collection(collection_name=COLLECTION_NAME)

            print("Collection deleted.")

        print(f"Creating collection '{COLLECTION_NAME}'...")

        self.create_collection()

        print("Collection ready.")

    # ==============================================================
    # Payload construction
    # ==============================================================

    @staticmethod
    def _build_payload(chunk: PatentChunk) -> dict:
        """
        Build the Qdrant payload dict for a PatentChunk.

        Centralised here to avoid duplicating the field list
        between insert() and insert_batch().
        """

        return {
            "patent_id": chunk.patent_id,
            "section": chunk.section,
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "section_chunk_index": chunk.section_chunk_index,
            "document_chunk_index": chunk.document_chunk_index,
            "total_chunks": chunk.total_chunks,
            "token_count": chunk.token_count,
            "word_count": chunk.word_count,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "metadata": chunk.metadata,
        }

    # ==============================================================
    # Insert
    # ==============================================================

    def insert(self, chunk: PatentChunk):
        """
        Insert a single chunk.
        """

        point = PointStruct(
            id=chunk.point_id,
            vector=chunk.vector,
            payload=self._build_payload(chunk),
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point],
        )

    def insert_batch(self, chunks: list[PatentChunk]):
        """
        Insert multiple chunks in one request.
        """

        points = [
            PointStruct(
                id=chunk.point_id,
                vector=chunk.vector,
                payload=self._build_payload(chunk),
            )
            for chunk in chunks
        ]

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(f"Inserted {len(points)} chunks.")

    def search(
        self,
        query_vector: list[float],
        score_threshold: float = 0.30   ,
        limit: int = VECTOR_TOP_K,
    ):
        """
        Search similar vectors.
        """

        return self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            score_threshold=score_threshold,
            limit=limit,
        )

    def count_points(self):
        """
        Return total indexed vectors.
        """

        result = self.client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )

        return result.count

    def get_collection_info(self):
        """
        Return collection information.
        """

        return self.client.get_collection(COLLECTION_NAME)

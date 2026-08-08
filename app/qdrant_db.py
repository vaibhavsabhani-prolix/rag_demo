"""
Qdrant Database Layer

Two collections back the pipeline:

    patent_chunks  - searchable chunk data + embedding vector.
                     One point per chunk. No patent metadata.

    patents        - patent metadata JSON, stored once per patent.
                     No vectors, looked up by patent_id.

Splitting storage this way means metadata is written once per patent
instead of once per chunk, which matters at 180M-patent scale where a
single patent can produce dozens of chunks.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import (
    CHUNKS_COLLECTION_NAME,
    PATENTS_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_SIZE,
    VECTOR_TOP_K,
)

from app.filter_engine import FilterEngine
from app.models.patent_chunk import PatentChunk
from app.query_understanding.models import MetadataFilter

# Points in the "patents" collection are keyed by patent_id, but Qdrant
# point IDs must be an unsigned int or a UUID. This namespace makes the
# patent_id -> point_id mapping deterministic, so re-ingesting a patent
# overwrites its existing metadata point instead of duplicating it.
_PATENT_POINT_NAMESPACE = uuid.UUID("6f6d3b2e-6b8b-4b1a-9c1a-8f6e2f6b8b1a")


def _patent_point_id(patent_id: str) -> str:
    return str(uuid.uuid5(_PATENT_POINT_NAMESPACE, patent_id))


class QdrantDB:

    def __init__(self):

        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

    # ==============================================================
    # Collection management
    # ==============================================================

    def create_collections(self):
        """
        Create both collections if they don't already exist.
        """

        self._create_chunks_collection()
        self._create_patents_collection()

    def _create_chunks_collection(self):

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if CHUNKS_COLLECTION_NAME in names:
            print(f"Collection '{CHUNKS_COLLECTION_NAME}' already exists.")
            return

        self.client.create_collection(
            collection_name=CHUNKS_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(f"Collection '{CHUNKS_COLLECTION_NAME}' created.")

    def _create_patents_collection(self):

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if PATENTS_COLLECTION_NAME in names:
            print(f"Collection '{PATENTS_COLLECTION_NAME}' already exists.")
            return

        # No vectors — this collection only stores metadata payloads,
        # looked up directly by point ID.
        self.client.create_collection(
            collection_name=PATENTS_COLLECTION_NAME,
            vectors_config={},
        )

        print(f"Collection '{PATENTS_COLLECTION_NAME}' created.")

    def reset_collections(self):
        """
        Delete and recreate both collections.
        """

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        for name in (CHUNKS_COLLECTION_NAME, PATENTS_COLLECTION_NAME):
            if name in names:
                print(f"Deleting collection '{name}'...")
                self.client.delete_collection(collection_name=name)
                print("Collection deleted.")

        print("Recreating collections...")

        self.create_collections()

        print("Collections ready.")

    # ==============================================================
    # Chunk payload construction
    # ==============================================================

    @staticmethod
    def _build_chunk_payload(chunk: PatentChunk) -> dict:
        """
        Build the Qdrant payload dict for a PatentChunk.

        Contains only searchable chunk data — patent metadata lives
        in the "patents" collection and is looked up by patent_id.
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
        }

    # ==============================================================
    # Chunk insert
    # ==============================================================

    def insert(self, chunk: PatentChunk):
        """
        Insert a single chunk.
        """

        point = PointStruct(
            id=chunk.point_id,
            vector=chunk.vector,
            payload=self._build_chunk_payload(chunk),
        )

        self.client.upsert(
            collection_name=CHUNKS_COLLECTION_NAME,
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
                payload=self._build_chunk_payload(chunk),
            )
            for chunk in chunks
        ]

        self.client.upsert(
            collection_name=CHUNKS_COLLECTION_NAME,
            points=points,
        )

        print(f"Inserted {len(points)} chunks.")

    # ==============================================================
    # Patent metadata
    # ==============================================================

    def upsert_patent_metadata(self, patent_id: str, metadata: dict):
        """
        Store a patent's metadata once, keyed by patent_id.

        Safe to call once per patent per ingestion run — re-ingesting
        the same patent overwrites its existing metadata point rather
        than duplicating it.
        """

        point = PointStruct(
            id=_patent_point_id(patent_id),
            vector={},
            payload={
                "patent_id": patent_id,
                "metadata": metadata,
            },
        )

        self.client.upsert(
            collection_name=PATENTS_COLLECTION_NAME,
            points=[point],
        )

    def get_patents_metadata(self, patent_ids: list[str]) -> dict[str, dict]:
        """
        Fetch metadata for a list of patent_ids in one request.

        Returns a dict mapping patent_id -> metadata. Patent IDs with
        no stored metadata are omitted from the result.
        """

        if not patent_ids:
            return {}

        unique_ids = list(dict.fromkeys(patent_ids))

        records = self.client.retrieve(
            collection_name=PATENTS_COLLECTION_NAME,
            ids=[_patent_point_id(pid) for pid in unique_ids],
            with_payload=True,
        )

        return {
            record.payload["patent_id"]: record.payload.get("metadata", {})
            for record in records
            if record.payload
        }

    # ==============================================================
    # Metadata-first filtering (DEPRECATED)
    #
    # The active pipeline uses post-vector metadata filtering
    # (SemanticSearch._filter_candidates_by_metadata) instead.
    # This method is retained for backward compatibility but is
    # NOT called in the active search flow.
    # ==============================================================

    def filter_patent_ids(
        self,
        filters: list[MetadataFilter],
    ) -> list[str]:
        """
        **DEPRECATED** — Not used in the active search pipeline.

        The active architecture performs metadata filtering AFTER vector
        search (see SemanticSearch._filter_candidates_by_metadata).

        This method scrolls the entire 'patents' collection to find
        patent_ids satisfying *filters*. Retained for backward
        compatibility only.
        """

        if not filters:
            return []

        native_filters, python_filters = FilterEngine.split_native_and_python(filters)
        qdrant_filter = FilterEngine.to_qdrant_filter(native_filters)
        payload_fields = ["patent_id", "metadata"] if python_filters else ["patent_id"]

        matched_ids: list[str] = []
        next_offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=PATENTS_COLLECTION_NAME,
                scroll_filter=qdrant_filter,
                limit=256,
                offset=next_offset,
                with_payload=payload_fields,
            )

            if not records:
                break

            for record in records:
                if not record.payload:
                    continue

                if python_filters:
                    metadata = record.payload.get("metadata", {})
                    if not FilterEngine.matches(metadata, python_filters):
                        continue

                matched_ids.append(record.payload["patent_id"])

            if next_offset is None:
                break

        return matched_ids


    # ==============================================================
    # Search
    # ==============================================================

    def search(
        self,
        query_vector: list[float],
        score_threshold: float = 0.30,
        limit: int = VECTOR_TOP_K,
    ):
        """
        Search similar vectors in the chunks collection.

        Pure semantic vector search - no metadata filter involved. Any
        metadata-constraint narrowing happens afterward, in Python,
        against the patent_ids present in the returned candidates (see
        SemanticSearch._filter_candidates_by_metadata) - not here.
        """

        return self.client.search(
            collection_name=CHUNKS_COLLECTION_NAME,
            query_vector=query_vector,
            score_threshold=score_threshold,
            limit=limit,
        )

    # ==============================================================
    # Stats
    # ==============================================================

    def count_points(self):
        """
        Return total indexed chunk vectors.
        """

        result = self.client.count(
            collection_name=CHUNKS_COLLECTION_NAME,
            exact=True,
        )

        return result.count

    def count_patents(self):
        """
        Return total patents with stored metadata.
        """

        result = self.client.count(
            collection_name=PATENTS_COLLECTION_NAME,
            exact=True,
        )

        return result.count

    def get_collection_info(self):
        """
        Return chunk collection information.
        """

        return self.client.get_collection(CHUNKS_COLLECTION_NAME)

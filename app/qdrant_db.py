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
    BATCH_SIZE,
    CHUNKS_COLLECTION_NAME,
    PATENTS_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    VECTOR_SIZE,
)

from app.models.patent_chunk import PatentChunk

# Points in the "patents" collection are keyed by patent_id, but Qdrant
# point IDs must be an unsigned int or a UUID. This namespace makes the
# patent_id -> point_id mapping deterministic, so re-ingesting a patent
# overwrites its existing metadata point instead of duplicating it.
_PATENT_POINT_NAMESPACE = uuid.UUID("6f6d3b2e-6b8b-4b1a-9c1a-8f6e2f6b8b1a")


def _patent_point_id(patent_id: str) -> str:
    return str(uuid.uuid5(_PATENT_POINT_NAMESPACE, patent_id))


class QdrantDB:
    def __init__(self, timeout: float = QDRANT_TIMEOUT):

        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=timeout,
        )

    def ensure_payload_index(self):
        """Create keyword payload index on patent_id if it does not exist."""
        try:
            self.client.create_payload_index(
                collection_name=CHUNKS_COLLECTION_NAME,
                field_name="patent_id",
                field_schema="keyword",
            )
        except Exception:
            pass

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

        try:
            self.client.create_payload_index(
                collection_name=CHUNKS_COLLECTION_NAME,
                field_name="patent_id",
                field_schema="keyword",
            )
        except Exception:
            pass

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

        self._create_chunks_collection()
        self._create_patents_collection()

        print("Collections ready.")

    # ==============================================================
    # Chunk insertion
    # ==============================================================

    @staticmethod
    def _build_chunk_payload(chunk: PatentChunk) -> dict:
        """
        Build the dictionary stored in a chunk point's payload.
        """
        return {
            "patent_id": chunk.patent_id,
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "word_count": chunk.word_count,
            "document_chunk_index": chunk.document_chunk_index,
            "section_chunk_index": chunk.section_chunk_index,
            "total_sections": chunk.total_sections,
            "section_total_chunks": chunk.section_total_chunks,
            "total_chunks": chunk.total_chunks,
            "chunk_uuid": chunk.chunk_uuid or chunk.point_id,
            "created_at": chunk.created_at,
        }

    def insert_batch(
        self,
        chunks: list[PatentChunk],
        batch_size: int = BATCH_SIZE,
        on_progress=None,
        wait: bool = True,
    ) -> int:
        """
        Upload embedded chunks to the chunks collection in batches.

        *on_progress*, if provided, is called with the number of points
        uploaded as each batch completes.
        """

        if not chunks:
            return 0

        inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            points = [
                PointStruct(
                    id=chunk.point_id,
                    vector=chunk.vector,
                    payload=self._build_chunk_payload(chunk),
                )
                for chunk in batch
            ]

            self.client.upsert(
                collection_name=CHUNKS_COLLECTION_NAME,
                points=points,
                wait=wait,
            )

            if on_progress is not None:
                on_progress(len(points))

            inserted += len(points)

        return inserted

    # ==============================================================
    # Patent metadata
    # ==============================================================

    def upsert_patent_metadata(self, patent_id: str, metadata: dict):
        """
        Store patent metadata in the "patents" collection.

        Point ID is a deterministic UUID derived from patent_id (see
        _patent_point_id), so re-running ingestion overwrites the
        existing record instead of inserting a duplicate point.
        """

        self.client.upsert(
            collection_name=PATENTS_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=_patent_point_id(patent_id),
                    vector={},
                    payload={
                        "patent_id": patent_id,
                        "metadata": metadata,
                    },
                )
            ],
        )

    def upsert_patent_metadata_batch(
        self,
        patents: list[tuple[str, dict]],
        batch_size: int = 64,
        wait: bool = True,
    ) -> int:
        """
        Store metadata for multiple patents in the "patents" collection
        in batches of *batch_size*, saving per-patent HTTP round trips.

        *patents* is a list of (patent_id, metadata_dict) tuples.
        Returns the total number of patent records written.
        """

        if not patents:
            return 0

        inserted = 0

        for i in range(0, len(patents), batch_size):
            batch = patents[i : i + batch_size]

            points = [
                PointStruct(
                    id=_patent_point_id(patent_id),
                    vector={},
                    payload={
                        "patent_id": patent_id,
                        "metadata": metadata,
                    },
                )
                for patent_id, metadata in batch
            ]

            self.client.upsert(
                collection_name=PATENTS_COLLECTION_NAME,
                points=points,
                wait=wait,
            )

            inserted += len(points)

        return inserted

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

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

import json
import logging
import re
import uuid
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from app.config import (
    BATCH_SIZE,
    CHUNKS_COLLECTION_NAME,
    CHUNKS_COLLECTION_PREFIX,
    PATENTS_COLLECTION_NAME,
    PATENTS_COLLECTION_PREFIX,
    QDRANT_HOST,
    QDRANT_INDEXING_THRESHOLD_KB,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    VECTOR_SIZE,
)

from app.models.collection import SearchCollection
from app.models.patent_chunk import PatentChunk

logger = logging.getLogger(__name__)

# Points in the "patents" collection are keyed by patent_id, but Qdrant
# point IDs must be an unsigned int or a UUID. This namespace makes the
# patent_id -> point_id mapping deterministic, so re-ingesting a patent
# overwrites its existing metadata point instead of duplicating it.
_PATENT_POINT_NAMESPACE = uuid.UUID("6f6d3b2e-6b8b-4b1a-9c1a-8f6e2f6b8b1a")


def _patent_point_id(patent_id: str) -> str:
    return str(uuid.uuid5(_PATENT_POINT_NAMESPACE, patent_id))


def _natural_key(name: str) -> list:
    """Sort key that orders "512" before "2048" and "4096"."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


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
    # Bulk loading
    # ==============================================================

    def pause_indexing(self):
        """
        Stop Qdrant from building HNSW indexes for the chunks collection's new
        segments, so a big ingest doesn't compete with index builds. Segments
        already indexed stay indexed. Searches scan unindexed segments in full
        (slower) until resume_indexing().
        """
        self.client.update_collection(
            collection_name=CHUNKS_COLLECTION_NAME,
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )

    def resume_indexing(self):
        """
        Index everything added since pause_indexing(). Qdrant does it in the
        background; the collection is YELLOW until it's done.
        """
        self.client.update_collection(
            collection_name=CHUNKS_COLLECTION_NAME,
            optimizers_config=OptimizersConfigDiff(indexing_threshold=QDRANT_INDEXING_THRESHOLD_KB),
        )

    def index_status(self) -> tuple[str, int, int]:
        """The chunks collection's status (green/yellow/grey/red), indexed vectors and points."""
        info = self.client.get_collection(CHUNKS_COLLECTION_NAME)
        return info.status.value, info.indexed_vectors_count or 0, info.points_count or 0

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

    def get_patents_metadata(
        self,
        patent_ids: list[str],
        collection_name: str = PATENTS_COLLECTION_NAME,
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch metadata for a list of patent_ids in one request.

        Returns a dict mapping patent_id -> metadata. Patent IDs with
        no stored metadata are omitted from the result.
        """

        if not patent_ids:
            return {}

        unique_ids = list(dict.fromkeys(patent_ids))

        records = self.client.retrieve(
            collection_name=collection_name,
            ids=[_patent_point_id(pid) for pid in unique_ids],
            with_payload=True,
        )

        return {
            record.payload["patent_id"]: record.payload.get("metadata", {})
            for record in records
            if record.payload
        }

    # ==============================================================
    # Search collections
    # ==============================================================

    def list_search_collections(self) -> list[SearchCollection]:
        """
        Return every searchable collection pair in Qdrant, sorted by name
        (numbers in natural order, so "512" comes before "2048").

        A pair is "patent_chunks_<name>" plus "patents_metadata_<name>".
        A chunks collection without its metadata collection is skipped,
        since search needs both.
        """

        existing = {c.name for c in self.client.get_collections().collections}
        pairs = []

        for chunks_name in sorted(existing, key=_natural_key):
            if not chunks_name.startswith(CHUNKS_COLLECTION_PREFIX):
                continue
            name = chunks_name[len(CHUNKS_COLLECTION_PREFIX):]
            patents_name = PATENTS_COLLECTION_PREFIX + name
            if not name or patents_name not in existing:
                continue

            pairs.append(
                SearchCollection(
                    name=name,
                    chunks_collection=chunks_name,
                    patents_collection=patents_name,
                    chunk_count=self.client.get_collection(chunks_name).points_count or 0,
                    patent_count=self.client.get_collection(patents_name).points_count or 0,
                )
            )

        return pairs

    def get_collection_memory(self, collection_name: str) -> dict:
        """
        Real disk/RAM usage for *collection_name*, from Qdrant's
        GET /collections/{name}/memory endpoint. Not wrapped by qdrant-client,
        so called directly over HTTP. Returns zeros if the call fails (e.g. an
        older Qdrant build without this route).
        """
        try:
            resp = requests.get(
                f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/memory",
                timeout=QDRANT_TIMEOUT,
            )
            resp.raise_for_status()
            total = resp.json()["result"]["total"]
            return {
                "disk_bytes": total.get("disk_bytes", 0) or 0,
                "ram_bytes": total.get("ram_bytes", 0) or 0,
            }
        except Exception as exc:
            logger.warning("Could not read memory usage for collection %s: %s", collection_name, exc)
            return {"disk_bytes": 0, "ram_bytes": 0}

    def find_common_patent_ids(
        self,
        collections: list[SearchCollection],
        limit: int = 2,
        scan_limit: int = 50,
    ) -> list[str]:
        """
        Patent IDs present in every given collection's patents metadata
        collection, so the same patent's chunk footprint can be compared
        across chunk-size variants (e.g. 512 / 2048 / 4096).

        Scrolls a page of candidates from the first collection, then checks
        each against the rest by deterministic point ID lookup (cheap -
        no scrolling) until *limit* common patent IDs are found.
        """
        if not collections:
            return []

        first, *rest = collections
        results, _ = self.client.scroll(
            collection_name=first.patents_collection,
            limit=scan_limit,
            with_payload=True,
        )
        candidate_ids = [
            pt.payload.get("patent_id")
            for pt in results
            if pt.payload and pt.payload.get("patent_id")
        ]

        common: list[str] = []
        for pid in candidate_ids:
            if all(self.get_patents_metadata([pid], c.patents_collection) for c in rest):
                common.append(pid)
            if len(common) >= limit:
                break
        return common

    def get_patent_chunk_storage(self, patent_id: str, chunks_collection: str) -> dict:
        """
        Chunk count and an estimated storage footprint for *patent_id*'s own
        chunks in *chunks_collection*.

        Vector bytes are exact (one float32 per vector dimension, per
        chunk). Payload bytes are the actual JSON-serialized payload size
        (dominated by chunk text, which is why 512-token chunking produces
        more total vector overhead than 4096 for the same patent, even
        though the underlying text is identical). This is a raw-data
        estimate, not Qdrant's true on-disk size - see get_collection_memory
        for the real, collection-level disk/RAM usage, which also includes
        HNSW graph, WAL and segment overhead this estimate leaves out.
        """
        chunk_count = 0
        payload_bytes = 0
        next_offset = None
        patent_filter = Filter(must=[FieldCondition(key="patent_id", match=MatchValue(value=patent_id))])

        while True:
            points, next_offset = self.client.scroll(
                collection_name=chunks_collection,
                scroll_filter=patent_filter,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                chunk_count += 1
                if pt.payload:
                    payload_bytes += len(json.dumps(pt.payload, default=str).encode("utf-8"))
            if next_offset is None:
                break

        vector_bytes = chunk_count * VECTOR_SIZE * 4
        return {
            "chunk_count": chunk_count,
            "vector_bytes": vector_bytes,
            "payload_bytes": payload_bytes,
            "estimated_total_bytes": vector_bytes + payload_bytes,
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

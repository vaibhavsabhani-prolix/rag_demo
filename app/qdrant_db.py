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

import types
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from app.config import (
    CANDIDATE_CHUNKS_PER_PATENT,
    CHUNKS_COLLECTION_NAME,
    PATENT_CANDIDATE_TOP_K,
    PATENTS_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    VECTOR_SIZE,
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

    def insert_batch(self, chunks: list[PatentChunk], wait: bool = True):
        """
        Insert multiple chunks in one request.

        *wait* controls whether Qdrant acknowledges only after the points
        are committed to the index. Bulk ingestion passes wait=False so
        the next batch can be embedded while Qdrant indexes this one -
        the points are still accepted and durably queued, they just are
        not guaranteed searchable the instant this returns. Callers that
        read straight back (tests, single-shot inserts) keep the default.
        """

        if not chunks:
            return 0

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
            wait=wait,
        )

        return len(points)

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

    def upsert_patent_metadata_batch(
        self,
        entries: list[tuple[str, dict]],
        wait: bool = True,
    ):
        """
        Store metadata for many patents in one request.

        Same semantics as upsert_patent_metadata() - deterministic point
        IDs, so re-ingesting overwrites rather than duplicates - but one
        HTTP round trip for the whole group instead of one per patent,
        which is what ingestion needs at directory scale.
        """

        if not entries:
            return 0

        points = [
            PointStruct(
                id=_patent_point_id(patent_id),
                vector={},
                payload={
                    "patent_id": patent_id,
                    "metadata": metadata,
                },
            )
            for patent_id, metadata in entries
        ]

        self.client.upsert(
            collection_name=PATENTS_COLLECTION_NAME,
            points=points,
            wait=wait,
        )

        return len(points)

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
    # Metadata-first filtering
    #
    # Used when a query has metadata_filters but no real semantic
    # content to vector-search with (see ParsedQuery.is_metadata_only) -
    # post-vector-search filtering (SemanticSearch._filter_patent_ids_by_metadata)
    # can only ever match patents within the PATENT_CANDIDATE_TOP_K candidate pool,
    # which is the wrong tool when the query is purely a metadata
    # lookup ("applications filed in 2008 by Wyeth").
    # ==============================================================

    def filter_patent_ids(
        self,
        filters: list[MetadataFilter],
    ) -> list[str]:

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

    def get_chunks_for_patent_ids(self, patent_ids: list[str]) -> list:
        """
        Fetch every chunk belonging to *patent_ids* directly from the
        chunks collection, via a native Qdrant filter on the safe
        "patent_id" field - no vector search involved.

        Companion to filter_patent_ids() for metadata-only queries:
        once the matching patent_ids are known, this retrieves their
        full chunk set (unbounded by PATENT_CANDIDATE_TOP_K) so the
        existing reranker/aggregation code can run unchanged.
        """

        if not patent_ids:
            return []

        scroll_filter = Filter(
            must=[FieldCondition(key="patent_id", match=MatchAny(any=patent_ids))]
        )

        chunks: list = []
        next_offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=CHUNKS_COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=256,
                offset=next_offset,
                with_payload=True,
            )

            # Scroll returns plain Records (no .score - there was no
            # ranking involved). Wrap as ScoredPoint-like objects so the
            # existing reranker/aggregation/diagnostics code, which all
            # expect a `.score` alongside `.id`/`.payload`, works
            # unchanged. score=1.0 marks "confirmed metadata match",
            # not a similarity value.
            chunks.extend(
                types.SimpleNamespace(id=record.id, score=1.0, payload=record.payload)
                for record in records
            )

            if next_offset is None:
                break

        return chunks

    # ==============================================================
    # Search
    # ==============================================================

    def search(
        self,
        query_vector: list[float],
        score_threshold: float = 0.30,
        limit: int = PATENT_CANDIDATE_TOP_K,
    ):
        """
        Identify the top *limit* distinct candidate PATENTS via semantic
        vector search, grouped by patent_id.

        Pure semantic vector search - no metadata filter involved. Any
        metadata-constraint narrowing happens afterward, in Python,
        against the patent_ids present in the returned candidates (see
        SemanticSearch._filter_patent_ids_by_metadata) - not here.

        Uses Qdrant's group-by search with group_size=CANDIDATE_CHUNKS_PER_PATENT
        so *limit* bounds the number of distinct PATENTS returned, not
        chunks - a single patent with many similar-scoring chunks can't
        crowd other relevant patents out of the candidate pool the way a
        flat top-K chunk search could. Only each patent's top
        CANDIDATE_CHUNKS_PER_PATENT best-matching chunks are returned
        here, capping reranker cost/latency per patent - callers that
        need every chunk of a candidate patent (e.g. the metadata-only
        path) should use get_chunks_for_patent_ids() instead, which
        fetches a patent's full, unbounded chunk set.

        NOTE: the returned list is flattened across groups (`hit for
        group in result.groups for hit in group.hits`), so the SAME
        patent_id can appear up to CANDIDATE_CHUNKS_PER_PATENT times in
        it - this is intentional here (callers that need the full
        per-patent chunk pool, e.g. for reranking, want that), but a
        caller that needs a patent-level, one-row-per-patent view must
        collapse duplicates itself (see
        app.semantic_search._dedupe_top_chunk_per_patent) rather than
        assume this method already returns unique patent_ids.
        """

        result = self.client.query_points_groups(
            collection_name=CHUNKS_COLLECTION_NAME,
            query=query_vector,
            group_by="patent_id",
            limit=limit,
            group_size=CANDIDATE_CHUNKS_PER_PATENT,
            score_threshold=score_threshold,
            timeout=int(QDRANT_TIMEOUT),
        )

        return [hit for group in result.groups for hit in group.hits]

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

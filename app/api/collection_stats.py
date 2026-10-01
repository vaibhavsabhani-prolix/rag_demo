"""
Collection storage stats endpoints.

GET /stats reports, per searchable collection (one per chunk-size variant,
e.g. 512 / 2048 / 4096), how many chunks and patents it holds and how much
disk/RAM Qdrant actually reports for it - plus the same 1-2 sample patents'
chunk count and estimated storage footprint in each collection, so the
effect of chunk size on a single patent is visible side by side.

GET /patent/{patent_id} runs that same per-collection lookup on demand for
any patent_id the user supplies.
"""

import logging

from fastapi import APIRouter, Path

from app.api.schemas import (
    CollectionMemoryStats,
    CollectionsOverview,
    CollectionStorageStats,
    PatentLookupResult,
    SamplePatentCollectionChunks,
    SamplePatentStats,
)
from app.config import VECTOR_SIZE
from app.models.collection import SearchCollection
from app.qdrant_db import QdrantDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["collections"])

SAMPLE_PATENT_COUNT = 2

# Preferred example patents for the sample comparison, used ahead of
# auto-discovered ones when present in every collection. Falls back to
# QdrantDB.find_common_patent_ids() to fill any remaining slots.
PINNED_SAMPLE_PATENT_IDS = ["JP2025106030A"]


def _max_chunk_tokens(name: str) -> int | None:
    return int(name) if name.isdigit() else None


def _patent_storage_across_collections(
    db: QdrantDB, patent_id: str, collections: list[SearchCollection]
) -> tuple[dict, list[SamplePatentCollectionChunks], list[str]]:
    """
    Look up *patent_id* in every given collection. Returns (metadata, the
    per-collection chunk/storage breakdown for collections it was found in,
    and the names of collections it has no metadata in at all).
    """
    metadata: dict = {}
    per_collection: list[SamplePatentCollectionChunks] = []
    missing: list[str] = []

    for c in collections:
        found_meta = db.get_patents_metadata([patent_id], c.patents_collection)
        if not found_meta:
            missing.append(c.name)
            continue
        metadata = metadata or found_meta.get(patent_id, {})
        try:
            chunk_stats = db.get_patent_chunk_storage(patent_id, c.chunks_collection)
        except Exception:
            logger.exception("Could not compute chunk storage for %s in %s", patent_id, c.chunks_collection)
            continue
        per_collection.append(SamplePatentCollectionChunks(collection=c.name, **chunk_stats))

    return metadata, per_collection, missing


def _resolve_sample_patent_ids(db: QdrantDB, collections: list, limit: int) -> list[str]:
    chosen: list[str] = []
    for pid in PINNED_SAMPLE_PATENT_IDS:
        if len(chosen) >= limit:
            break
        if all(db.get_patents_metadata([pid], c.patents_collection) for c in collections):
            chosen.append(pid)
    if len(chosen) < limit:
        for pid in db.find_common_patent_ids(collections, limit=limit):
            if pid not in chosen:
                chosen.append(pid)
            if len(chosen) >= limit:
                break
    return chosen


@router.get("/stats", response_model=CollectionsOverview)
def get_collections_overview() -> CollectionsOverview:
    db = QdrantDB()
    collections = db.list_search_collections()

    collection_stats: list[CollectionStorageStats] = []
    for c in collections:
        chunks_mem = db.get_collection_memory(c.chunks_collection)
        patents_mem = db.get_collection_memory(c.patents_collection)
        collection_stats.append(
            CollectionStorageStats(
                name=c.name,
                chunks_collection=c.chunks_collection,
                patents_collection=c.patents_collection,
                chunk_count=c.chunk_count,
                patent_count=c.patent_count,
                max_chunk_tokens=_max_chunk_tokens(c.name),
                chunks_memory=CollectionMemoryStats(**chunks_mem),
                patents_memory=CollectionMemoryStats(**patents_mem),
                total_disk_bytes=chunks_mem["disk_bytes"] + patents_mem["disk_bytes"],
                total_ram_bytes=chunks_mem["ram_bytes"] + patents_mem["ram_bytes"],
            )
        )

    sample_patents: list[SamplePatentStats] = []
    if collections:
        patent_ids = _resolve_sample_patent_ids(db, collections, SAMPLE_PATENT_COUNT)
        for pid in patent_ids:
            metadata, per_collection, _missing = _patent_storage_across_collections(db, pid, collections)
            sample_patents.append(
                SamplePatentStats(patent_id=pid, metadata=metadata, per_collection=per_collection)
            )

    return CollectionsOverview(
        vector_size=VECTOR_SIZE,
        collections=collection_stats,
        sample_patents=sample_patents,
    )


@router.get("/patent/{patent_id}", response_model=PatentLookupResult)
def get_patent_storage(
    patent_id: str = Path(..., min_length=1, max_length=64),
) -> PatentLookupResult:
    """
    Look up a user-supplied patent_id across every searchable collection and
    report the same chunk count / estimated storage breakdown as the pinned
    sample patents, so any patent can be inspected on demand.
    """
    db = QdrantDB()
    collections = db.list_search_collections()
    patent_id = patent_id.strip()

    metadata, per_collection, missing = _patent_storage_across_collections(db, patent_id, collections)

    return PatentLookupResult(
        patent_id=patent_id,
        found=bool(per_collection),
        metadata=metadata,
        per_collection=per_collection,
        missing_collections=missing,
    )

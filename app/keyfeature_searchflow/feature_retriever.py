import logging
from typing import Any

from qdrant_client.models import QueryRequest

from app.config import (
    CHUNKS_COLLECTION_NAME,
    PATENTS_COLLECTION_NAME,
)
from app.keyfeature_searchflow.models import (
    FeatureChunkResult,
    FeatureSearchResult,
    KeyFeatureItem,
)
from app.models.collection import SearchCollection
from app.qdrant_db import QdrantDB

logger = logging.getLogger(__name__)


class KeyFeatureRetriever:
    def __init__(self, db: QdrantDB | None = None):
        self.db = db or QdrantDB()

    def _resolve_collection_names(
        self, collection: SearchCollection | str
    ) -> tuple[str, str]:
        if isinstance(collection, SearchCollection):
            return collection.chunks_collection, collection.patents_collection

        coll_str = str(collection).strip()
        if coll_str.startswith("patent_chunks_"):
            name = coll_str.removeprefix("patent_chunks_")
            return coll_str, f"patents_metadata_{name}"
        elif coll_str.startswith("patents_metadata_"):
            name = coll_str.removeprefix("patents_metadata_")
            return f"patent_chunks_{name}", coll_str
        elif coll_str:
            return f"patent_chunks_{coll_str}", f"patents_metadata_{coll_str}"
        return CHUNKS_COLLECTION_NAME, PATENTS_COLLECTION_NAME

    def retrieve_for_features(
        self,
        features_with_embeddings: list[tuple[KeyFeatureItem, list[float]]],
        collection: SearchCollection | str = CHUNKS_COLLECTION_NAME,
        score_threshold: float = 0.5,
        limit: int = 2000,
    ) -> list[FeatureSearchResult]:

        if not features_with_embeddings:
            return []

        chunks_coll, patents_coll = self._resolve_collection_names(collection)

        # Build one QueryRequest per feature
        # Each query uses ONLY that feature's own embedding vector and filters by score_threshold
        requests = [
            QueryRequest(
                query=embedding,
                score_threshold=score_threshold,
                limit=limit,
                with_payload=True,
            )
            for _, embedding in features_with_embeddings
        ]

        # Execute batched queries in Qdrant (1:1 with features)
        batch_results = self.db.client.query_batch_points(
            collection_name=chunks_coll,
            requests=requests,
        )

        # Collect all patent IDs to enrich chunks with patent metadata (e.g. title)
        all_patent_ids: list[str] = []
        hits_per_feature: list[list[Any]] = []

        for search_res in batch_results:
            hits_list: list[Any] = list(search_res.points) if search_res.points else []
            hits_per_feature.append(hits_list)
            for hit in hits_list:
                payload: dict[str, Any] = hit.payload or {}
                pid = payload.get("patent_id")
                if pid:
                    all_patent_ids.append(pid)

        # Fetch patent metadata in batch
        metadata_map = {}
        if all_patent_ids:
            try:
                metadata_map = self.db.get_patents_metadata(
                    patent_ids=list(set(all_patent_ids)),
                    collection_name=patents_coll,
                )
            except Exception as exc:  # noqa: BLE001 - metadata enrichment is best-effort
                logger.warning("Could not fetch patent metadata: %s", exc)

        # Build independent results per feature, preserving feature_id
        results: list[FeatureSearchResult] = []
        for (feature_item, _), hits in zip(features_with_embeddings, hits_per_feature):
            chunk_results: list[FeatureChunkResult] = []
            for hit in hits:
                score = float(hit.score)
                if score < score_threshold:
                    continue
                payload: dict[str, Any] = hit.payload or {}
                patent_id = payload.get("patent_id", "")
                chunk_id = int(payload.get("chunk_id", 0))
                text = payload.get("text", "")
                section = payload.get("section")
                point_id = str(hit.id)
                document_chunk_index = payload.get("document_chunk_index")
                token_count = payload.get("token_count")

                pat_meta = metadata_map.get(patent_id, {})
                title = pat_meta.get("title") or payload.get("title")

                chunk_results.append(
                    FeatureChunkResult(
                        chunk_id=chunk_id,
                        patent_id=patent_id,
                        score=score,
                        text=text,
                        section=section,
                        title=title,
                        point_id=point_id,
                        document_chunk_index=document_chunk_index,
                        token_count=token_count,
                        metadata=pat_meta,
                    )
                )

            results.append(
                FeatureSearchResult(
                    feature_id=feature_item.feature_id,
                    feature=feature_item.feature,
                    chunks=chunk_results,
                )
            )

        return results

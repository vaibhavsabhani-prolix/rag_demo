import logging
from typing import Any

from qdrant_client.models import QueryRequest

from app.config import (
    CHUNKS_COLLECTION_NAME,
    PATENTS_COLLECTION_NAME,
    QDRANT_TIMEOUT,
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

    def resolve_collection_names(
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

    # Alias for backward compatibility
    _resolve_collection_names = resolve_collection_names

    def retrieve_for_features(
        self,
        features_with_embeddings: list[tuple[KeyFeatureItem, list[float]]],
        collection: SearchCollection | str = CHUNKS_COLLECTION_NAME,
        score_threshold: float = 0.0,
        limit: int = 5000,
    ) -> list[FeatureSearchResult]:

        if not features_with_embeddings:
            return []

        chunks_coll, _ = self.resolve_collection_names(collection)

        # Restrict payload retrieval strictly to fields needed for reranking & display.
        # This avoids reading and serializing unnecessary payload fields from disk.
        payload_fields = [
            "patent_id",
            "chunk_id",
            "text",
            "section",
            "document_chunk_index",
            "token_count",
            "title",
        ]

        # Build one QueryRequest per feature
        # Each query uses ONLY that feature's own embedding vector and optional score_threshold
        requests = [
            QueryRequest(
                query=embedding,
                score_threshold=score_threshold if score_threshold and score_threshold > 0 else None,
                limit=limit,
                with_payload=payload_fields,
            )
            for _, embedding in features_with_embeddings
        ]

        # Execute batched queries in Qdrant (1:1 with features)
        batch_results = self.db.client.query_batch_points(
            collection_name=chunks_coll,
            requests=requests,
            timeout=int(QDRANT_TIMEOUT),
        )

        hits_per_feature: list[list[Any]] = [
            list(search_res.points) if search_res.points else []
            for search_res in batch_results
        ]

        # Build independent results per feature, preserving feature_id.
        # NOTE: Patent metadata fetching is intentionally deferred to pipeline.py after reranking
        # to avoid fetching full metadata for thousands of candidate chunks that get pruned.
        results: list[FeatureSearchResult] = []
        for (feature_item, _), hits in zip(features_with_embeddings, hits_per_feature):
            chunk_results: list[FeatureChunkResult] = []
            for hit in hits:
                score = float(hit.score)
                if score_threshold and score_threshold > 0 and score < score_threshold:
                    continue
                payload: dict[str, Any] = hit.payload or {}
                patent_id = payload.get("patent_id", "")
                chunk_id = int(payload.get("chunk_id", 0))
                text = payload.get("text", "")
                section = payload.get("section")
                point_id = str(hit.id)
                document_chunk_index = payload.get("document_chunk_index")
                token_count = payload.get("token_count")
                title = payload.get("title")

                chunk_results.append(
                    FeatureChunkResult(
                        chunk_id=chunk_id,
                        patent_id=patent_id,
                        feature_id=feature_item.feature_id,
                        score=score,
                        retrieval_score=score,
                        reranker_score=None,
                        text=text,
                        section=section,
                        title=title,
                        point_id=point_id,
                        document_chunk_index=document_chunk_index,
                        token_count=token_count,
                        metadata={},
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

import logging
import time
from collections.abc import Callable
from typing import Any

from app.config import (
    CHUNKS_COLLECTION_NAME,
    KEYFEATURE_RERANK_TOP_K,
    KEYFEATURE_RETRIEVAL_LIMIT,
)
from app.keyfeature_searchflow.feature_embedder import KeyFeatureEmbedder
from app.keyfeature_searchflow.feature_extractor import KeyFeatureExtractor
from app.keyfeature_searchflow.feature_reranker import KeyFeatureReranker
from app.keyfeature_searchflow.feature_retriever import KeyFeatureRetriever
from app.keyfeature_searchflow.models import (
    FeatureSearchResult,
    KeyFeatureSearchResponse,
)
from app.models.collection import SearchCollection
from app.qdrant_db import QdrantDB

logger = logging.getLogger(__name__)


class KeyFeatureSearchPipeline:
    def __init__(
        self,
        extractor: KeyFeatureExtractor | None = None,
        embedder: KeyFeatureEmbedder | None = None,
        retriever: KeyFeatureRetriever | None = None,
        reranker: KeyFeatureReranker | None = None,
        db: QdrantDB | None = None,
    ):
        self.extractor = extractor or KeyFeatureExtractor()
        self.embedder = embedder or KeyFeatureEmbedder()
        self.retriever = retriever or KeyFeatureRetriever(db=db)
        self.reranker = reranker or KeyFeatureReranker()

    def _enrich_results_metadata(
        self,
        results: list[FeatureSearchResult],
        collection: SearchCollection | str,
    ) -> None:
        """Enrich only the final top-K winning chunks with full patent metadata.

        Deferred until after reranking so we only look up metadata for ~20-50 patents
        instead of thousands of candidate chunks.
        """
        unique_patent_ids = list(
            {
                chunk.patent_id
                for feat_res in results
                for chunk in feat_res.chunks
                if chunk.patent_id
            }
        )
        if not unique_patent_ids:
            return

        try:
            _, patents_coll = self.retriever.resolve_collection_names(collection)
            metadata_map = self.retriever.db.get_patents_metadata(
                patent_ids=unique_patent_ids,
                collection_name=patents_coll,
            )
            for feat_res in results:
                for chunk in feat_res.chunks:
                    if chunk.patent_id in metadata_map:
                        pat_meta = metadata_map[chunk.patent_id]
                        chunk.metadata = pat_meta
                        if pat_meta.get("title"):
                            chunk.title = pat_meta.get("title")
        except Exception as exc:  # noqa: BLE001 - metadata enrichment is best-effort
            logger.warning("Could not enrich patent metadata: %s", exc)

    def run(
        self,
        problem: str,
        invention_title: str,
        invention_details: str,
        collection: SearchCollection | str = CHUNKS_COLLECTION_NAME,
        score_threshold: float = 0.0,
        retrieval_limit: int = KEYFEATURE_RETRIEVAL_LIMIT,
        top_k: int = KEYFEATURE_RERANK_TOP_K,
        on_phase: Callable[[int, str, Any], None] | None = None,
    ) -> KeyFeatureSearchResponse:

        t_overall_start = time.perf_counter()

        # Step 1: Feature Extraction
        t0 = time.perf_counter()
        features = self.extractor.extract_features(
            problem,
            invention_title,
            invention_details,
        )
        elapsed_extract = (time.perf_counter() - t0) * 1000
        if on_phase:
            on_phase(1, "Feature Extraction", {"key_features": [f.model_dump() for f in features]})

        # Step 2: Independent Feature Embedding
        t1 = time.perf_counter()
        features_with_embeddings = self.embedder.embed_features(features)
        elapsed_embed = (time.perf_counter() - t1) * 1000
        if on_phase:
            on_phase(2, "Feature Embedding", {})

        # Step 3: Independent Feature Retrieval (initial candidate retrieval from Qdrant)
        t2 = time.perf_counter()
        results = self.retriever.retrieve_for_features(
            features_with_embeddings=features_with_embeddings,
            collection=collection,
            score_threshold=score_threshold,
            limit=retrieval_limit,
        )
        elapsed_retrieve = (time.perf_counter() - t2) * 1000
        if on_phase:
            on_phase(3, "Candidate Retrieval", {})

        # Step 4: Cross-Encoder Reranking & Top-K Selection
        t3 = time.perf_counter()
        results = self.reranker.rerank_feature_results(
            results=results,
            top_k=top_k,
        )
        elapsed_rerank = (time.perf_counter() - t3) * 1000

        # Step 5: Metadata Enrichment for winning chunks only
        t4 = time.perf_counter()
        self._enrich_results_metadata(results, collection)
        elapsed_enrich = (time.perf_counter() - t4) * 1000

        if on_phase:
            on_phase(4, "Cross-Encoder Reranking", {"results": [r.model_dump() for r in results]})

        total_elapsed = (time.perf_counter() - t_overall_start) * 1000
        collection_str = collection.name if isinstance(collection, SearchCollection) else str(collection)

        return KeyFeatureSearchResponse(
            problem=problem,
            invention_title=invention_title,
            invention_details=invention_details,
            collection=collection_str,
            score_threshold=score_threshold,
            retrieval_limit=retrieval_limit,
            top_k=top_k,
            total_features=len(features),
            results=results,
            timings={
                "extraction_ms": round(elapsed_extract, 2),
                "embedding_ms": round(elapsed_embed, 2),
                "retrieval_ms": round(elapsed_retrieve, 2),
                "reranking_ms": round(elapsed_rerank, 2),
                "metadata_enrich_ms": round(elapsed_enrich, 2),
                "total_ms": round(total_elapsed, 2),
            },
        )

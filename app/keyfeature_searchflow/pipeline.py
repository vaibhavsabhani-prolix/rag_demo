import logging
import time
from collections.abc import Callable
from typing import Any

from app.config import CHUNKS_COLLECTION_NAME
from app.keyfeature_searchflow.feature_embedder import KeyFeatureEmbedder
from app.keyfeature_searchflow.feature_extractor import KeyFeatureExtractor
from app.keyfeature_searchflow.feature_reranker import KeyFeatureReranker
from app.keyfeature_searchflow.feature_retriever import KeyFeatureRetriever
from app.keyfeature_searchflow.models import (
    FeatureSearchResult,
    KeyFeatureItem,
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
            | {
                p.patent_id
                for feat_res in results
                for p in getattr(feat_res, "vector_retrieval_patents", [])
                if p.patent_id and not p.title
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
                feat_id = feat_res.feature_id
                for chunk in feat_res.chunks:
                    chunk.feature_id = feat_id
                    if chunk.patent_id in metadata_map:
                        pat_meta = metadata_map[chunk.patent_id]
                        chunk.metadata = pat_meta
                        if pat_meta.get("title"):
                            chunk.title = pat_meta.get("title")
                for pat in feat_res.patents:
                    if pat.patent_id in metadata_map:
                        pat_meta = metadata_map[pat.patent_id]
                        pat.metadata = pat_meta
                        if pat_meta.get("title"):
                            pat.title = pat_meta.get("title")
                    for pch in pat.chunks:
                        pch.feature_id = feat_id
                        if pch.patent_id in metadata_map:
                            pat_meta = metadata_map[pch.patent_id]
                            pch.metadata = pat_meta
                            if pat_meta.get("title"):
                                pch.title = pat_meta.get("title")
                for p in getattr(feat_res, "vector_retrieval_patents", []):
                    if p.patent_id in metadata_map and not p.title:
                        p.title = metadata_map[p.patent_id].get("title")
                    for pch in getattr(p, "chunks", []):
                        pch.feature_id = feat_id
                for p in getattr(feat_res, "rerank_filtered_patents", []):
                    if p.patent_id in metadata_map and not p.title:
                        p.title = metadata_map[p.patent_id].get("title")
                    for pch in getattr(p, "chunks", []):
                        pch.feature_id = feat_id
                for p in getattr(feat_res, "topk_pruned_patents", []):
                    if p.patent_id in metadata_map and not p.title:
                        p.title = metadata_map[p.patent_id].get("title")
                    for pch in getattr(p, "chunks", []):
                        pch.feature_id = feat_id
        except Exception as exc:  # noqa: BLE001 - metadata enrichment is best-effort
            logger.warning("Could not enrich patent metadata: %s", exc)

    def extract_features(
        self,
        problem: str,
        invention_title: str,
        invention_details: str,
    ) -> list[KeyFeatureItem]:
        return self.extractor.extract_features(
            problem,
            invention_title,
            invention_details,
        )

    def embed_features(
        self,
        features: list[KeyFeatureItem],
    ) -> list[Any]:
        return self.embedder.embed_features(features)

    def run_collection(
        self,
        problem: str,
        invention_title: str,
        invention_details: str,
        features: list[KeyFeatureItem],
        features_with_embeddings: list[Any],
        collection: SearchCollection | str = CHUNKS_COLLECTION_NAME,
        score_threshold: float = 0.0,
        rerank_score_threshold: float = 0.6,
        retrieval_limit: int = 5000,
        top_k: int = 20,
        on_phase: Callable[[int, str, Any], None] | None = None,
        extraction_ms: float = 0.0,
        embedding_ms: float = 0.0,
        overall_start_time: float | None = None,
    ) -> KeyFeatureSearchResponse:
        t_start = overall_start_time if overall_start_time is not None else time.perf_counter()

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
        # Emitted before the (expensive) rerank call so the UI reflects this
        # stage while it's actually running, not only once it has finished.
        if on_phase:
            on_phase(4, "Cross-Encoder Reranking", {})

        def on_feature_progress(done: int, total: int, feature_text: str) -> None:
            if on_phase:
                on_phase(
                    4,
                    f"Cross-Encoder Reranking ({done}/{total} features) — {feature_text[:60]}",
                    {},
                )

        t3 = time.perf_counter()
        chunks_coll, _ = self.retriever.resolve_collection_names(collection)
        results = self.reranker.rerank_feature_results(
            results=results,
            top_k=top_k,
            rerank_score_threshold=rerank_score_threshold,
            chunks_collection=chunks_coll,
            on_feature_progress=on_feature_progress,
        )
        elapsed_rerank = (time.perf_counter() - t3) * 1000

        # Step 5: Metadata Enrichment for winning chunks only
        t4 = time.perf_counter()
        self._enrich_results_metadata(results, collection)
        elapsed_enrich = (time.perf_counter() - t4) * 1000

        if on_phase:
            on_phase(4, "Cross-Encoder Reranking", {"results": [r.model_dump() for r in results]})

        total_elapsed = (time.perf_counter() - t_start) * 1000
        collection_str = collection.name if isinstance(collection, SearchCollection) else str(collection)

        return KeyFeatureSearchResponse(
            problem=problem,
            invention_title=invention_title,
            invention_details=invention_details,
            collection=collection_str,
            score_threshold=score_threshold,
            rerank_score_threshold=rerank_score_threshold,
            retrieval_limit=retrieval_limit,
            top_k=top_k,
            total_features=len(features),
            results=results,
            timings={
                "extraction_ms": round(extraction_ms, 2),
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(elapsed_retrieve, 2),
                "reranking_ms": round(elapsed_rerank, 2),
                "metadata_enrich_ms": round(elapsed_enrich, 2),
                "collection_total_ms": round(elapsed_retrieve + elapsed_rerank + elapsed_enrich, 2),
                "total_ms": round(total_elapsed, 2),
            },
        )

    def run(
        self,
        problem: str,
        invention_title: str,
        invention_details: str,
        collection: SearchCollection | str = CHUNKS_COLLECTION_NAME,
        score_threshold: float = 0.0,
        rerank_score_threshold: float = 0.6,
        retrieval_limit: int = 5000,
        top_k: int = 20,
        on_phase: Callable[[int, str, Any], None] | None = None,
    ) -> KeyFeatureSearchResponse:
        t_overall_start = time.perf_counter()

        # Step 1: Feature Extraction
        t0 = time.perf_counter()
        features = self.extract_features(
            problem,
            invention_title,
            invention_details,
        )
        elapsed_extract = (time.perf_counter() - t0) * 1000
        if on_phase:
            on_phase(1, "Feature Extraction", {"key_features": [f.model_dump() for f in features]})

        # Step 2: Independent Feature Embedding
        t1 = time.perf_counter()
        features_with_embeddings = self.embed_features(features)
        elapsed_embed = (time.perf_counter() - t1) * 1000
        if on_phase:
            on_phase(2, "Feature Embedding", {})

        return self.run_collection(
            problem=problem,
            invention_title=invention_title,
            invention_details=invention_details,
            features=features,
            features_with_embeddings=features_with_embeddings,
            collection=collection,
            score_threshold=score_threshold,
            rerank_score_threshold=rerank_score_threshold,
            retrieval_limit=retrieval_limit,
            top_k=top_k,
            on_phase=on_phase,
            extraction_ms=elapsed_extract,
            embedding_ms=elapsed_embed,
            overall_start_time=t_overall_start,
        )

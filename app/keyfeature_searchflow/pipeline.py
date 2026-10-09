import logging
import time

from app.config import CHUNKS_COLLECTION_NAME
from app.keyfeature_searchflow.feature_embedder import KeyFeatureEmbedder
from app.keyfeature_searchflow.feature_extractor import KeyFeatureExtractor
from app.keyfeature_searchflow.feature_retriever import KeyFeatureRetriever
from app.keyfeature_searchflow.models import (
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
        db: QdrantDB | None = None,
    ):
        self.extractor = extractor or KeyFeatureExtractor()
        self.embedder = embedder or KeyFeatureEmbedder()
        self.retriever = retriever or KeyFeatureRetriever(db=db)

    def run(
        self,
        problem: str,
        invention_title: str,
        invention_details: str,
        collection: SearchCollection | str = CHUNKS_COLLECTION_NAME,
        top_k: int = 10,
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

        # Step 2: Independent Feature Embedding
        t1 = time.perf_counter()
        features_with_embeddings = self.embedder.embed_features(features)
        elapsed_embed = (time.perf_counter() - t1) * 1000

        # Step 3: Independent Feature Retrieval
        t2 = time.perf_counter()
        results = self.retriever.retrieve_for_features(
            features_with_embeddings=features_with_embeddings,
            collection=collection,
            candidate_top_k=top_k,
        )
        elapsed_retrieve = (time.perf_counter() - t2) * 1000

        total_elapsed = (time.perf_counter() - t_overall_start) * 1000
        collection_str = collection.name if isinstance(collection, SearchCollection) else str(collection)

        return KeyFeatureSearchResponse(
            problem=problem,
            invention_title=invention_title,
            invention_details=invention_details,
            collection=collection_str,
            total_features=len(features),
            results=results,
            timings={
                "extraction_ms": round(elapsed_extract, 2),
                "embedding_ms": round(elapsed_embed, 2),
                "retrieval_ms": round(elapsed_retrieve, 2),
                "total_ms": round(total_elapsed, 2),
            },
        )


"""
Key Feature Search Flow package.

Simple 3-step pipeline:
1. Extract 5-15 key features from (problem, title, details) using LLM.
2. Independently embed each key feature.
3. Independently search Qdrant for each feature, preserving feature_id association.
"""

from app.keyfeature_searchflow.feature_embedder import KeyFeatureEmbedder
from app.keyfeature_searchflow.feature_extractor import KeyFeatureExtractor
from app.keyfeature_searchflow.feature_retriever import KeyFeatureRetriever
from app.keyfeature_searchflow.models import (
    FeatureChunkResult,
    FeatureExtractionResponse,
    FeatureSearchResult,
    KeyFeatureItem,
    KeyFeatureSearchRequest,
    KeyFeatureSearchResponse,
)
from app.keyfeature_searchflow.pipeline import KeyFeatureSearchPipeline
from app.keyfeature_searchflow.prompts import build_feature_extraction_prompt

__all__ = [
    "FeatureChunkResult",
    "FeatureExtractionResponse",
    "FeatureSearchResult",
    "KeyFeatureEmbedder",
    "KeyFeatureExtractor",
    "KeyFeatureItem",
    "KeyFeatureRetriever",
    "KeyFeatureSearchPipeline",
    "KeyFeatureSearchRequest",
    "KeyFeatureSearchResponse",
    "build_feature_extraction_prompt",
]


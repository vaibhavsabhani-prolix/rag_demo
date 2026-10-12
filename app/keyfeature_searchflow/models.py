"""
Data models for the Key Feature Search Flow.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.config import (
    KEYFEATURE_RERANK_TOP_K,
    KEYFEATURE_RETRIEVAL_LIMIT,
)
from app.models.reranking import ChunkHighlight


class KeyFeatureItem(BaseModel):
    feature_id: int
    feature: str


class FeatureExtractionResponse(BaseModel):
    key_features: list[KeyFeatureItem]


class FeatureChunkResult(BaseModel):
    chunk_id: int
    patent_id: str
    score: float
    retrieval_score: float | None = None
    reranker_score: float | None = None
    text: str
    section: str | None = None
    title: str | None = None
    point_id: str | None = None
    document_chunk_index: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlight: ChunkHighlight | None = None


class FeatureSearchResult(BaseModel):
    feature_id: int
    feature: str
    chunks: list[FeatureChunkResult] = Field(default_factory=list[FeatureChunkResult])


class KeyFeatureSearchRequest(BaseModel):
    problem: str
    invention_title: str
    invention_details: str
    collection: str | None = None
    score_threshold: float = 0.0
    top_k: int = KEYFEATURE_RERANK_TOP_K
    retrieval_limit: int = KEYFEATURE_RETRIEVAL_LIMIT


class KeyFeatureSearchResponse(BaseModel):
    problem: str
    invention_title: str
    invention_details: str
    collection: str
    score_threshold: float | None = None
    top_k: int = KEYFEATURE_RERANK_TOP_K
    retrieval_limit: int = KEYFEATURE_RETRIEVAL_LIMIT
    total_features: int
    results: list[FeatureSearchResult]
    timings: dict[str, float] = Field(default_factory=dict)


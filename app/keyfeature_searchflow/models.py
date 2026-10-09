"""
Data models for the Key Feature Search Flow.
"""

from typing import Any

from pydantic import BaseModel, Field


class KeyFeatureItem(BaseModel):
    feature_id: int
    feature: str


class FeatureExtractionResponse(BaseModel):
    key_features: list[KeyFeatureItem]


class FeatureChunkResult(BaseModel):
    chunk_id: int
    patent_id: str
    score: float
    text: str
    section: str | None = None
    title: str | None = None
    point_id: str | None = None
    document_chunk_index: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureSearchResult(BaseModel):
    feature_id: int
    feature: str
    chunks: list[FeatureChunkResult] = Field(default_factory=list[FeatureChunkResult])


class KeyFeatureSearchRequest(BaseModel):
    problem: str
    invention_title: str
    invention_details: str
    collection: str | None = None
    top_k: int = 10


class KeyFeatureSearchResponse(BaseModel):
    problem: str
    invention_title: str
    invention_details: str
    collection: str
    total_features: int
    results: list[FeatureSearchResult]
    timings: dict[str, float] = Field(default_factory=dict)


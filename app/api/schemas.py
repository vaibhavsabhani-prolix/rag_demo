"""
API request/response schemas for the FastAPI search service.

Phase results are streamed as the pipeline's own Pydantic models (dumped to
JSON), so only the envelope types live here.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    use_cache: bool = True


class CacheStats(BaseModel):
    size: int
    max_size: int
    hits: int
    misses: int
    hit_ratio: float


class PipelineConfig(BaseModel):
    query_llm_model: str
    query_llm_base_url: str
    embedding_model: str
    reranker_model: str
    chunks_collection: str
    patents_collection: str
    retrieval_top_k_per_view: int
    patent_candidate_top_k: int
    evidence_neighbor_chunks: int
    rerank_batch_size: int
    reranker_max_context_tokens: int
    final_score_threshold: float
    weights: Dict[str, float]
    patent_view_url_template: str


class HistoryTopResult(BaseModel):
    patent_id: str
    final_score: float
    title: str = ""


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query: str
    use_cache: bool
    cache_hit: bool
    status: str
    error: Optional[str]
    total_ms: Optional[float]
    result_count: int
    top_results: List[HistoryTopResult]
    created_at: datetime


class HistoryList(BaseModel):
    items: List[HistoryItem]
    total: int


class HistoryDetail(HistoryItem):
    # {"1": {"name", "elapsed_ms", "data"}, ...}
    phases: Dict[str, Any]


class UiSettings(BaseModel):
    """Appearance settings (shared by everyone using this server)."""

    # "system" follows the operating system's light/dark preference.
    theme: Literal["system", "light", "dark"] = "system"

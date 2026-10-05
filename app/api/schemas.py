"""
API request/response schemas for the FastAPI search service.

Phase results are streamed as the pipeline's own Pydantic models (dumped to
JSON), so only the envelope types live here.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.collection import SearchCollection


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    # Collection name from GET /api/collections; the default collection if omitted.
    collection: Optional[str] = Field(None, max_length=128)
    use_cache: bool = True


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    # Collection names to compare, run in this order; every collection if omitted or empty.
    collections: Optional[List[str]] = Field(None, max_length=16)


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
    retrieval_top_k: int
    rerank_batch_size: int
    reranker_max_context_tokens: int
    final_score_threshold: float
    weights: Dict[str, float]
    patent_view_url_template: str


class CollectionList(BaseModel):
    collections: List[SearchCollection]
    # Preselected in the UI; null when Qdrant has no searchable collection.
    default: Optional[str]


class HistoryTopResult(BaseModel):
    patent_id: str
    # None for a metadata-only query, which isn't scored (see FinalPatentResult.final_score).
    final_score: Optional[float] = None
    title: str = ""


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query: str
    collection: Optional[str]
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


class CollectionMemoryStats(BaseModel):
    disk_bytes: int
    ram_bytes: int


class CollectionStorageStats(BaseModel):
    name: str
    chunks_collection: str
    patents_collection: str
    chunk_count: int
    patent_count: int
    # Parsed from the collection name when it's numeric (e.g. "512" -> 512); null otherwise.
    max_chunk_tokens: Optional[int]
    chunks_memory: CollectionMemoryStats
    patents_memory: CollectionMemoryStats
    total_disk_bytes: int
    total_ram_bytes: int


class SamplePatentCollectionChunks(BaseModel):
    collection: str
    chunk_count: int
    vector_bytes: int
    payload_bytes: int
    estimated_total_bytes: int


class SamplePatentStats(BaseModel):
    patent_id: str
    metadata: Dict[str, Any]
    per_collection: List[SamplePatentCollectionChunks]


class CollectionsOverview(BaseModel):
    vector_size: int
    collections: List[CollectionStorageStats]
    # Same 1-2 patents, present in every collection, so their chunk count and
    # storage footprint can be compared head-to-head across chunk sizes.
    sample_patents: List[SamplePatentStats]


class PatentLookupResult(BaseModel):
    """On-demand version of SamplePatentStats for a user-supplied patent_id."""

    patent_id: str
    # False only when the patent_id has no metadata in any searchable collection.
    found: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    per_collection: List[SamplePatentCollectionChunks] = Field(default_factory=list)
    # Collection names the patent_id has no metadata in (so wasn't looked up there).
    missing_collections: List[str] = Field(default_factory=list)


class UiSettings(BaseModel):
    """Appearance settings (shared by everyone using this server)."""

    # "system" follows the operating system's light/dark preference.
    theme: Literal["system", "light", "dark"] = "system"
    # Colour of query-match highlights in evidence chunks, as #rrggbb.
    highlight_color: str = Field("#f59e0b", pattern=r"^#[0-9a-fA-F]{6}$")

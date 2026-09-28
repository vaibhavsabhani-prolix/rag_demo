"""
Patent Semantic Search — FastAPI service

Exposes the 7-phase SearchPipeline to the React frontend:

    GET    /api/config   pipeline settings and scoring weights
    GET    /api/cache    query-understanding LRU cache stats
    DELETE /api/cache    clear the LRU cache
    POST   /api/search   run the pipeline, streaming NDJSON events
    /api/history/...     saved searches (see app/api/history.py)
    /api/settings/ui     appearance settings (see app/api/settings.py)

Every search is saved to PostgreSQL with all of its phase results.

The search stream emits one JSON object per line:

    {"type": "start", "query": ..., "cache_hit": bool, "search_id": int | null}
    {"type": "phase", "phase": 1..7, "name": ..., "elapsed_ms": ..., "data": {...}}
    {"type": "done", "elapsed_ms": ...}
    {"type": "error", "message": ...}

Run with:  PYTHONPATH=. .venv/bin/uvicorn app.api.main:app --port 8000
"""

import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.history import router as history_router
from app.api.settings import router as settings_router
from app.api.schemas import CacheStats, PipelineConfig, SearchRequest
from app.config import (
    CHUNKS_COLLECTION_NAME,
    EVIDENCE_NEIGHBOR_CHUNKS,
    FINAL_SCORE_THRESHOLD,
    FINAL_WEIGHT_RELATIONSHIP,
    FINAL_WEIGHT_REQUIREMENT,
    FINAL_WEIGHT_RERANKER,
    FINAL_WEIGHT_RETRIEVAL,
    PATENT_CANDIDATE_TOP_K,
    PATENT_VIEW_URL_TEMPLATE,
    PATENTS_COLLECTION_NAME,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    RERANK_BATCH_SIZE,
    RERANKER_MAX_CONTEXT_TOKENS,
    RERANKER_REMOTE_MODEL,
    RETRIEVAL_TOP_K_PER_VIEW,
)
from app.db import repository
from app.db.database import init_db
from app.query_understanding.engine import QueryUnderstandingEngine
from app.reranking.reranker import BGEReranker
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.retriever import CandidateRetriever
from app.scoring.scorer import FinalScorer
from app.semantic_search import SearchPipeline
from app.verification.verifier import RelationshipVerifier

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# Fields left out of the stream because they duplicate other fields
# (Phase 4's evidence_by_patent repeats every chunk in patent_evidence_list).
STREAM_EXCLUDE = {4: {"evidence_by_patent"}}

pipeline: SearchPipeline


def build_pipeline() -> SearchPipeline:
    engine = QueryUnderstandingEngine()
    engine.warm_up()
    reranker = BGEReranker()
    return SearchPipeline(
        engine=engine,
        retriever=CandidateRetriever(),
        evidence_retriever=EvidenceRetriever(),
        verifier=RelationshipVerifier(),
        reranker=reranker,
        scorer=FinalScorer(),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    init_db()
    pipeline = build_pipeline()
    yield


app = FastAPI(title="Patent Semantic Search API", lifespan=lifespan)

# The Vite dev server proxies /api, but allow direct calls during development too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(history_router)
app.include_router(settings_router)


@app.get("/api/config", response_model=PipelineConfig)
def get_config() -> PipelineConfig:
    return PipelineConfig(
        query_llm_model=QUERY_LLM_REMOTE_MODEL,
        query_llm_base_url=QUERY_LLM_REMOTE_BASE_URL,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reranker_model=RERANKER_REMOTE_MODEL,
        chunks_collection=CHUNKS_COLLECTION_NAME,
        patents_collection=PATENTS_COLLECTION_NAME,
        retrieval_top_k_per_view=RETRIEVAL_TOP_K_PER_VIEW,
        patent_candidate_top_k=PATENT_CANDIDATE_TOP_K,
        evidence_neighbor_chunks=EVIDENCE_NEIGHBOR_CHUNKS,
        rerank_batch_size=RERANK_BATCH_SIZE,
        reranker_max_context_tokens=RERANKER_MAX_CONTEXT_TOKENS,
        final_score_threshold=FINAL_SCORE_THRESHOLD,
        weights={
            "relationship": FINAL_WEIGHT_RELATIONSHIP,
            "requirement": FINAL_WEIGHT_REQUIREMENT,
            "reranker": FINAL_WEIGHT_RERANKER,
            "retrieval": FINAL_WEIGHT_RETRIEVAL,
        },
        patent_view_url_template=PATENT_VIEW_URL_TEMPLATE,
    )


@app.get("/api/cache", response_model=CacheStats)
def get_cache_stats() -> CacheStats:
    return CacheStats(**pipeline.engine.cache.stats)


@app.delete("/api/cache", response_model=CacheStats)
def clear_cache() -> CacheStats:
    pipeline.engine.cache.clear()
    return CacheStats(**pipeline.engine.cache.stats)


def _line(event: dict) -> str:
    return json.dumps(event, default=str) + "\n"


def _save_start(query: str, use_cache: bool, cache_hit: bool) -> Optional[int]:
    """Record the search; history is best-effort and never blocks a search."""
    try:
        return repository.create_search(query, use_cache, cache_hit)
    except Exception:
        logger.exception("Could not save search to history")
        return None


def _save_finish(search_id: Optional[int], phases: dict, total_ms: float, error: Optional[str]) -> None:
    if search_id is None:
        return
    try:
        repository.finish_search(search_id, phases, total_ms, error)
    except Exception:
        logger.exception("Could not update search %s in history", search_id)


def _stream_search(req: SearchRequest) -> Iterator[str]:
    """
    Run the blocking pipeline on a worker thread and yield each phase's
    result as an NDJSON line the moment the pipeline reports it.
    """
    query = req.query.strip()
    cache_hit = req.use_cache and pipeline.engine.cache.get(query) is not None
    search_id = _save_start(query, req.use_cache, cache_hit)
    events: "queue.Queue[dict | None]" = queue.Queue()
    phases: dict[str, Any] = {}

    def on_phase_complete(phase_num: int, phase_name: str, result: object, elapsed_ms: float) -> None:
        data = (
            result.model_dump(mode="json", exclude=STREAM_EXCLUDE.get(phase_num))
            if isinstance(result, BaseModel)
            else result
        )
        phases[str(phase_num)] = {"name": phase_name, "elapsed_ms": elapsed_ms, "data": data}
        events.put({
            "type": "phase",
            "phase": phase_num,
            "name": phase_name,
            "elapsed_ms": elapsed_ms,
            "data": data,
        })

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            pipeline.run(query, use_cache=req.use_cache, on_phase_complete=on_phase_complete)
            final_event = {"type": "done", "elapsed_ms": (time.perf_counter() - t0) * 1000}
            error = None
        except Exception as exc:  # surface pipeline failures to the client
            error = f"{type(exc).__name__}: {exc}"
            final_event = {"type": "error", "message": error}
        # Save before the final event so the history list is current when the client refetches it.
        _save_finish(search_id, phases, (time.perf_counter() - t0) * 1000, error)
        events.put(final_event)
        events.put(None)

    yield _line({"type": "start", "query": query, "cache_hit": cache_hit, "search_id": search_id})
    threading.Thread(target=worker, daemon=True).start()
    while (event := events.get()) is not None:
        yield _line(event)


@app.post("/api/search")
def search(req: SearchRequest) -> StreamingResponse:
    return StreamingResponse(_stream_search(req), media_type="application/x-ndjson")


# Serve the built React app (frontend/dist) when present, e.g. after `npm run build`.
# Unknown paths fall back to index.html so client-side routes like /history/12 work on reload.
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        file = (FRONTEND_DIST / path).resolve()
        if path and file.is_file() and FRONTEND_DIST in file.parents:
            return FileResponse(file)
        return FileResponse(FRONTEND_DIST / "index.html")

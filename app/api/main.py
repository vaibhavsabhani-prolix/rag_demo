"""
Patent Semantic Search — FastAPI service

Exposes the 7-phase SearchPipeline to the React frontend:

    GET    /api/config   pipeline settings and scoring weights
    GET    /api/collections  Qdrant collections the user can search
    GET    /api/collections/stats  per-collection storage/RAM usage + sample-patent comparison
    GET    /api/cache    query-understanding LRU cache stats
    DELETE /api/cache    clear the LRU cache
    POST   /api/search   run the pipeline, streaming NDJSON events
    POST   /api/compare  run one query against several collections, streaming NDJSON events
    /api/history/...     saved searches (see app/api/history.py)
    /api/settings/ui     appearance settings (see app/api/settings.py)

Every search is saved to PostgreSQL with all of its phase results.

The search stream emits one JSON object per line:

    {"type": "start", "query": ..., "collection": ..., "cache_hit": bool, "search_id": int | null}
    {"type": "phase", "phase": 1..7, "name": ..., "elapsed_ms": ..., "data": {...}}
    {"type": "done", "elapsed_ms": ...}
    {"type": "error", "message": ...}

/api/compare parses the query once, then runs Phases 2-7 per collection, one
collection at a time so their timings are comparable. Its stream:

    {"type": "start", "query": ..., "collections": [...]}
    {"type": "phase", "collection": null, "phase": 1, ...}      shared by all collections
    {"type": "collection_start", "collection": ...}
    {"type": "phase", "collection": ..., "phase": 2..7, ..., "memory": {"start_bytes", "peak_bytes"} | null}
    {"type": "collection_done", "collection": ..., "elapsed_ms": ...,
     "memory": {"start_bytes", "peak_bytes", "end_bytes"} | null}
      or {"type": "collection_error", "collection": ..., "message": ...}
    {"type": "done", "elapsed_ms": ...}  or  {"type": "error", "message": ...}

"memory" is this process's resident memory (see app/api/memory.py); null where
it can't be read. Send a single collection to run them one by one.

Compare runs are not saved to history.

Run with:  PYTHONPATH=. .venv/bin/uvicorn app.api.main:app --port 8000
"""

import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.collection_stats import router as collection_stats_router
from app.api.history import router as history_router
from app.api.memory import MemorySampler, release_free_memory
from app.api.settings import router as settings_router
from app.api.schemas import (
    CacheStats,
    CollectionList,
    CompareRequest,
    PipelineConfig,
    SearchRequest,
)
from app.config import (
    CHUNKS_COLLECTION_NAME,
    EVIDENCE_NEIGHBOR_CHUNKS,
    FINAL_SCORE_THRESHOLD,
    FINAL_WEIGHT_RELATIONSHIP,
    FINAL_WEIGHT_REQUIREMENT,
    FINAL_WEIGHT_RERANKER,
    FINAL_WEIGHT_RETRIEVAL,
    PATENT_VIEW_URL_TEMPLATE,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    RERANK_BATCH_SIZE,
    RERANKER_MAX_CONTEXT_TOKENS,
    RERANKER_REMOTE_MODEL,
    RETRIEVAL_TOP_K_CHUNKS,
)
from app.db import repository
from app.db.database import init_db
from app.models.collection import SearchCollection
from app.qdrant_db import QdrantDB
from app.query_understanding.engine import QueryUnderstandingEngine
from app.reranking.reranker import BGEReranker
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.retriever import CandidateRetriever
from app.scoring.scorer import FinalScorer
from app.semantic_search import PHASE_NAMES, SearchPipeline
from app.verification.verifier import RelationshipVerifier

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
# Set in development (docker-compose.override.yml): UI pages redirect to the
# Vite dev server instead of serving the frontend built into the image.
FRONTEND_DEV_URL = os.getenv("FRONTEND_DEV_URL")

# Fields left out of the stream because they duplicate other fields
# (Phase 4's evidence_by_patent repeats every chunk in patent_evidence_list).
STREAM_EXCLUDE = {4: {"evidence_by_patent"}}

pipeline: SearchPipeline
db: QdrantDB


def build_pipeline(db: QdrantDB) -> SearchPipeline:
    engine = QueryUnderstandingEngine()
    engine.warm_up()
    reranker = BGEReranker()
    reranker.warm_up()
    return SearchPipeline(
        engine=engine,
        retriever=CandidateRetriever(db=db),
        evidence_retriever=EvidenceRetriever(db=db),
        verifier=RelationshipVerifier(),
        reranker=reranker,
        scorer=FinalScorer(),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline, db
    init_db()
    db = QdrantDB()
    pipeline = build_pipeline(db)
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
app.include_router(collection_stats_router)


@app.get("/api/config", response_model=PipelineConfig)
def get_config() -> PipelineConfig:
    return PipelineConfig(
        query_llm_model=QUERY_LLM_REMOTE_MODEL,
        query_llm_base_url=QUERY_LLM_REMOTE_BASE_URL,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reranker_model=RERANKER_REMOTE_MODEL,
        retrieval_top_k=RETRIEVAL_TOP_K_CHUNKS,
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


def _list_collections() -> list[SearchCollection]:
    try:
        return db.list_search_collections()
    except Exception as exc:
        logger.exception("Could not list Qdrant collections")
        raise HTTPException(status_code=503, detail=f"Could not reach Qdrant: {exc}") from exc


def _default_collection(collections: list[SearchCollection]) -> Optional[SearchCollection]:
    """The collection ingestion writes to, or else the first one."""
    for c in collections:
        if c.chunks_collection == CHUNKS_COLLECTION_NAME:
            return c
    return collections[0] if collections else None


@app.get("/api/collections", response_model=CollectionList)
def get_collections() -> CollectionList:
    collections = _list_collections()
    default = _default_collection(collections)
    return CollectionList(collections=collections, default=default.name if default else None)


def _resolve_collection(name: Optional[str]) -> SearchCollection:
    collections = _list_collections()
    if name is None:
        collection = _default_collection(collections)
        if collection is None:
            raise HTTPException(status_code=409, detail="Qdrant has no searchable collection.")
        return collection
    for c in collections:
        if c.name == name:
            return c
    raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")


@app.get("/api/cache", response_model=CacheStats)
def get_cache_stats() -> CacheStats:
    return CacheStats(**pipeline.engine.cache.stats)


@app.delete("/api/cache", response_model=CacheStats)
def clear_cache() -> CacheStats:
    pipeline.engine.cache.clear()
    return CacheStats(**pipeline.engine.cache.stats)


def _line(event: dict) -> str:
    return json.dumps(event, default=str) + "\n"


def _save_start(query: str, collection: str, use_cache: bool, cache_hit: bool) -> Optional[int]:
    """Record the search; history is best-effort and never blocks a search."""
    try:
        return repository.create_search(query, collection, use_cache, cache_hit)
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


def _phase_event(phase_num: int, phase_name: str, result: object, elapsed_ms: float) -> dict:
    data = (
        result.model_dump(mode="json", exclude=STREAM_EXCLUDE.get(phase_num))
        if isinstance(result, BaseModel)
        else result
    )
    return {
        "type": "phase",
        "phase": phase_num,
        "name": phase_name,
        "elapsed_ms": elapsed_ms,
        "data": data,
    }


def _stream_search(req: SearchRequest, collection: SearchCollection) -> Iterator[str]:
    """
    Run the blocking pipeline on a worker thread and yield each phase's
    result as an NDJSON line the moment the pipeline reports it.
    """
    query = req.query.strip()
    cache_hit = req.use_cache and pipeline.engine.cache.get(query) is not None
    search_id = _save_start(query, collection.name, req.use_cache, cache_hit)
    events: "queue.Queue[dict | None]" = queue.Queue()
    phases: dict[str, Any] = {}

    def on_phase_complete(phase_num: int, phase_name: str, result: object, elapsed_ms: float) -> None:
        event = _phase_event(phase_num, phase_name, result, elapsed_ms)
        phases[str(phase_num)] = {"name": phase_name, "elapsed_ms": elapsed_ms, "data": event["data"]}
        events.put(event)

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            pipeline.run(
                query,
                collection,
                use_cache=req.use_cache,
                on_phase_complete=on_phase_complete,
            )
            final_event = {"type": "done", "elapsed_ms": (time.perf_counter() - t0) * 1000}
            error = None
        except Exception as exc:  # surface pipeline failures to the client
            error = f"{type(exc).__name__}: {exc}"
            final_event = {"type": "error", "message": error}
        # Save before the final event so the history list is current when the client refetches it.
        _save_finish(search_id, phases, (time.perf_counter() - t0) * 1000, error)
        events.put(final_event)
        events.put(None)

    yield _line({
        "type": "start",
        "query": query,
        "collection": collection.name,
        "cache_hit": cache_hit,
        "search_id": search_id,
    })
    threading.Thread(target=worker, daemon=True).start()
    while (event := events.get()) is not None:
        yield _line(event)


@app.post("/api/search")
def search(req: SearchRequest) -> StreamingResponse:
    # Resolved before streaming so an unknown collection is a plain HTTP error.
    collection = _resolve_collection(req.collection)
    return StreamingResponse(_stream_search(req, collection), media_type="application/x-ndjson")


def _resolve_collections(names: Optional[list[str]]) -> list[SearchCollection]:
    collections = _list_collections()
    if not names:
        if not collections:
            raise HTTPException(status_code=409, detail="Qdrant has no searchable collection.")
        return collections
    by_name = {c.name: c for c in collections}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise HTTPException(status_code=404, detail=f"Collection(s) not found: {', '.join(missing)}")
    return [by_name[n] for n in dict.fromkeys(names)]


def _stream_compare(req: CompareRequest, collections: list[SearchCollection]) -> Iterator[str]:
    """
    Parse the query once, then run Phases 2-7 for each collection in turn on a
    worker thread, yielding each event as it happens. A collection that fails
    is reported and the rest still run.
    """
    query = req.query.strip()
    events: "queue.Queue[dict | None]" = queue.Queue()
    # Set when the client goes away, so the remaining collections are skipped.
    cancelled = threading.Event()

    def phase_callback(
        collection: Optional[str], memory: Optional[MemorySampler] = None
    ) -> Callable[[int, str, object, float], None]:
        def on_phase_complete(phase_num: int, phase_name: str, result: object, elapsed_ms: float) -> None:
            usage = memory.end_lap() if memory else None
            events.put({
                **_phase_event(phase_num, phase_name, result, elapsed_ms),
                "collection": collection,
                "memory": usage,
            })
            # The next phase's lap starts after this one's result is serialized.
            if memory:
                memory.start_lap()

        return on_phase_complete

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            t1 = time.perf_counter()
            parsed_query = pipeline.engine.parse(query, use_cache=False)
            phase_callback(None)(1, PHASE_NAMES[1], parsed_query, (time.perf_counter() - t1) * 1000)

            for collection in collections:
                if cancelled.is_set():
                    break
                events.put({"type": "collection_start", "collection": collection.name})
                # Start every collection from the same baseline; not part of its timing.
                release_free_memory()
                t_col = time.perf_counter()
                try:
                    with MemorySampler() as memory:
                        pipeline.run_parsed(parsed_query, collection, phase_callback(collection.name, memory))
                    events.put({
                        "type": "collection_done",
                        "collection": collection.name,
                        "elapsed_ms": (time.perf_counter() - t_col) * 1000,
                        "memory": memory.total(),
                    })
                except Exception as exc:
                    logger.exception("Compare run failed for collection %s", collection.name)
                    events.put({
                        "type": "collection_error",
                        "collection": collection.name,
                        "message": f"{type(exc).__name__}: {exc}",
                    })
            events.put({"type": "done", "elapsed_ms": (time.perf_counter() - t0) * 1000})
        except Exception as exc:  # Phase 1 failed, so no collection can run
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        events.put(None)

    yield _line({
        "type": "start",
        "query": query,
        "collections": [c.name for c in collections],
    })
    threading.Thread(target=worker, daemon=True).start()
    try:
        while (event := events.get()) is not None:
            yield _line(event)
    finally:
        cancelled.set()


@app.post("/api/compare")
def compare(req: CompareRequest) -> StreamingResponse:
    collections = _resolve_collections(req.collections)
    return StreamingResponse(_stream_compare(req, collections), media_type="application/x-ndjson")


# Serve the built React app (frontend/dist) when present, e.g. after `npm run build`.
# Unknown paths fall back to index.html so client-side routes like /history/12 work on reload.
if FRONTEND_DEV_URL:

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_dev(path: str, request: Request) -> RedirectResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"{FRONTEND_DEV_URL.rstrip('/')}/{path}{query}")

elif FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        file = (FRONTEND_DIST / path).resolve()
        if path and file.is_file() and FRONTEND_DIST in file.parents:
            return FileResponse(file)
        return FileResponse(FRONTEND_DIST / "index.html")

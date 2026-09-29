"""Read/write helpers for search history. Each call uses its own short session."""

from typing import Any, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import undefer

from app.db.database import SessionLocal
from app.db.models import AppSetting, SearchRecord

TOP_RESULTS_PREVIEW = 3


def _top_results(phases: dict[str, Any]) -> list[dict[str, Any]]:
    results = (phases.get("7") or {}).get("data", {}).get("results", [])
    preview = []
    for r in results[:TOP_RESULTS_PREVIEW]:
        meta = r.get("metadata") or {}
        preview.append({
            "patent_id": r["patent_id"],
            "final_score": r["final_score"],
            "title": meta.get("Title-english") or meta.get("Title") or "",
        })
    return preview


def create_search(query: str, collection: str, use_cache: bool, cache_hit: bool) -> int:
    with SessionLocal.begin() as session:
        record = SearchRecord(
            query=query, collection=collection, use_cache=use_cache, cache_hit=cache_hit
        )
        session.add(record)
        session.flush()
        return record.id


def finish_search(
    search_id: int,
    phases: dict[str, Any],
    total_ms: float,
    error: Optional[str] = None,
) -> None:
    results = (phases.get("7") or {}).get("data", {}).get("results", [])
    with SessionLocal.begin() as session:
        record = session.get(SearchRecord, search_id)
        if record is None:
            return
        record.status = "error" if error else "success"
        record.error = error
        record.total_ms = total_ms
        record.phases = phases
        record.result_count = len(results)
        record.top_results = _top_results(phases)


def list_searches(limit: int, offset: int, query: Optional[str] = None) -> tuple[list[SearchRecord], int]:
    stmt = select(SearchRecord)
    if query:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(SearchRecord.query.ilike(f"%{escaped}%", escape="\\"))
    with SessionLocal() as session:
        total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = session.scalars(
            stmt.order_by(SearchRecord.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return list(rows), total


def get_search(search_id: int) -> Optional[SearchRecord]:
    with SessionLocal() as session:
        return session.get(SearchRecord, search_id, options=[undefer(SearchRecord.phases)])


def delete_search(search_id: int) -> bool:
    with SessionLocal.begin() as session:
        return session.execute(delete(SearchRecord).where(SearchRecord.id == search_id)).rowcount > 0


def clear_searches() -> int:
    with SessionLocal.begin() as session:
        return session.execute(delete(SearchRecord)).rowcount


def get_setting(key: str) -> Optional[dict[str, Any]]:
    with SessionLocal() as session:
        row = session.get(AppSetting, key)
        return row.value if row else None


def set_setting(key: str, value: dict[str, Any]) -> None:
    with SessionLocal.begin() as session:
        session.merge(AppSetting(key=key, value=value))

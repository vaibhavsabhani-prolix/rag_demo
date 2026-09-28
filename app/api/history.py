"""Search history endpoints, backed by PostgreSQL."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import HistoryDetail, HistoryItem, HistoryList
from app.db import repository

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=HistoryList)
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, max_length=200),
) -> HistoryList:
    rows, total = repository.list_searches(limit=limit, offset=offset, query=q or None)
    return HistoryList(items=[HistoryItem.model_validate(r) for r in rows], total=total)


@router.get("/{search_id}", response_model=HistoryDetail)
def get_history(search_id: int) -> HistoryDetail:
    record = repository.get_search(search_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Search not found")
    return HistoryDetail.model_validate(record)


@router.delete("/{search_id}", status_code=204)
def delete_history(search_id: int) -> None:
    if not repository.delete_search(search_id):
        raise HTTPException(status_code=404, detail="Search not found")


@router.delete("", status_code=204)
def clear_history() -> None:
    repository.clear_searches()

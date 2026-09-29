"""Database tables."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, deferred, mapped_column

from app.db.database import Base


class SearchRecord(Base):
    """One run of the search pipeline, with every phase's result."""

    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    # Name of the searched collection pair, e.g. "2048". Null for searches
    # saved before the collection could be chosen.
    collection: Mapped[Optional[str]] = mapped_column(String(128))
    use_cache: Mapped[bool]
    cache_hit: Mapped[bool] = mapped_column(default=False)
    # "running" | "success" | "error"
    status: Mapped[str] = mapped_column(String(16), default="running")
    error: Mapped[Optional[str]] = mapped_column(Text)
    total_ms: Mapped[Optional[float]]
    result_count: Mapped[int] = mapped_column(default=0)
    # Small preview for the history list: [{patent_id, final_score, title}]
    top_results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    # {"1": {"name", "elapsed_ms", "data"}, ...} — large, so only loaded when asked for.
    phases: Mapped[dict[str, Any]] = deferred(mapped_column(JSONB, default=dict))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class AppSetting(Base):
    """Key/value application settings, e.g. "ui" -> {"theme": ...}."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

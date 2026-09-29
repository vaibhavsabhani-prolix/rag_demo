"""SQLAlchemy engine, session factory and table creation."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# create_all() only creates missing tables, so columns added to an existing
# table are added here. Each statement must be safe to run on every start.
_COLUMN_MIGRATIONS = [
    "ALTER TABLE search_history ADD COLUMN IF NOT EXISTS collection VARCHAR(128)",
]


def init_db() -> None:
    """Create any missing tables and columns."""
    from app.db import models  # noqa: F401  (registers the models on Base)

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for statement in _COLUMN_MIGRATIONS:
            conn.execute(text(statement))

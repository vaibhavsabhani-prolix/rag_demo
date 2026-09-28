"""SQLAlchemy engine, session factory and table creation."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create any missing tables."""
    from app.db import models  # noqa: F401  (registers the models on Base)

    Base.metadata.create_all(engine)

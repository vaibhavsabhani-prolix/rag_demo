"""
Patent Chunk Model

A single chunk of a patent document, ready for embedding and indexing.
All metadata fields have sensible defaults for backward compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class PatentChunk:
    """
    A single chunk of a patent document, ready for embedding and indexing.
    """

    # Automatically generate a unique UUID
    point_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    patent_id: str = ""

    chunk_id: int = 0

    text: str = ""

    metadata: dict = field(default_factory=dict)

    section: str = ""

    vector: list[float] = field(default_factory=list)

    # ---- Token / word counts ----

    token_count: int = 0

    word_count: int = 0

    # ---- Character offsets within the section ----

    start_offset: int = 0

    end_offset: int = 0

    # ---- Document-level metadata ----

    # Total chunks produced for this document
    total_chunks: int = 0

    # Position of this chunk within its section (1-based)
    section_chunk_index: int = 0

    # Position of this chunk within the document (1-based, same as chunk_id)
    document_chunk_index: int = 0

    # Total number of sections detected in the document
    total_sections: int = 0

    # Total chunks produced for this chunk's section
    section_total_chunks: int = 0

    # ---- Identifiers ----

    # Stable unique identifier (alias for point_id, included for clarity)
    chunk_uuid: str = ""

    # ISO 8601 creation timestamp
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
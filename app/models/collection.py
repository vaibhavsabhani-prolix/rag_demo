"""
A searchable collection: one chunks collection plus its patent metadata collection.
"""

from pydantic import BaseModel


class SearchCollection(BaseModel):
    # Shared suffix of the two Qdrant collections, e.g. "2048".
    name: str
    chunks_collection: str
    patents_collection: str
    chunk_count: int = 0
    patent_count: int = 0

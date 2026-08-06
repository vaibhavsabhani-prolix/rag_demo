from dataclasses import dataclass


@dataclass
class SearchResult:

    patent_id: str

    chunk_id: int

    score: float

    text: str

    metadata: dict
"""
Query match highlighting for evidence chunks (computed in Phase 5 reranking).

Exports:
- split_sentences / sentences_to_score: multilingual sentence splitter returning character spans
- extract_term_patterns / find_term_spans: query word matching
- build_chunk_highlight: combines sentence scores and term matches into a ChunkHighlight
"""

from app.highlighting.highlighter import (
    build_chunk_highlight,
    extract_term_patterns,
    find_term_spans,
    sentences_to_score,
    split_sentences,
)

__all__ = [
    "split_sentences",
    "sentences_to_score",
    "extract_term_patterns",
    "find_term_spans",
    "build_chunk_highlight",
]

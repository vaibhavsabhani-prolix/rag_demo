"""
Phase 2, Phase 3 & Phase 4: Candidate Retrieval, Metadata Filtering & Evidence Retrieval

Exports:
- CandidateRetriever: Main candidate retrieval engine
- build_retrieval_views: Dynamic retrieval view constructor
- build_qdrant_filter: Dynamic Qdrant filter constructor
- match_patent_metadata: Metadata constraint evaluator
- filter_candidates: Phase 3 metadata constraint enforcement
- matches_metadata_filter: Single filter evaluation
- matches_all_metadata_filters: Multi-filter AND evaluation
- EvidenceRetriever: Phase 4 bounded evidence retrieval engine
- build_evidence_query: Dynamic evidence query constructor
"""

from app.retrieval.evidence_retriever import (
    EvidenceRetriever,
    build_evidence_query,
)
from app.retrieval.filter_builder import (
    build_qdrant_filter,
    match_patent_metadata,
    resolve_payload_field_names,
)
from app.retrieval.metadata_filter import (
    filter_candidates,
    matches_all_metadata_filters,
    matches_metadata_filter,
)
from app.retrieval.retriever import CandidateRetriever
from app.retrieval.views import build_retrieval_views

__all__ = [
    "CandidateRetriever",
    "build_retrieval_views",
    "build_qdrant_filter",
    "match_patent_metadata",
    "resolve_payload_field_names",
    "filter_candidates",
    "matches_metadata_filter",
    "matches_all_metadata_filters",
    "EvidenceRetriever",
    "build_evidence_query",
]



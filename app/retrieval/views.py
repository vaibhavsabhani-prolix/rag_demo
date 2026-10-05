"""
Dynamic Retrieval Views Builder

Assembles the multiple retrieval views from a ParsedQuery. 'original' and
'semantic' are taken directly from the query; 'structured' is the LLM's own
dense retrieval-oriented synthesis of concepts, relationships, attributes,
and requirements (Query Understanding Phase 1) - no extra LLM call here.
"""

from typing import Dict
from app.models.parsed_query import ParsedQuery


def build_retrieval_views(parsed_query: ParsedQuery) -> Dict[str, str]:
    """
    Build the retrieval views from a ParsedQuery.

    Returns a mapping of view_name -> view_query_text:
      - 'original': raw user query
      - 'semantic': intent-preserving natural language query
      - 'structured': the LLM-provided dense retrieval synthesis
    """
    views: Dict[str, str] = {}

    orig = (parsed_query.original_query or "").strip()
    if orig:
        views["original"] = orig

    sem = (parsed_query.semantic_query or "").strip()
    if sem:
        views["semantic"] = sem
    elif orig:
        views["semantic"] = orig

    structured = (parsed_query.structured_query or "").strip()
    if structured:
        views["structured"] = structured
    elif sem:
        views["structured"] = sem
    elif orig:
        views["structured"] = orig

    return views

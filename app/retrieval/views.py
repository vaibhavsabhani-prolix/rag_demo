"""
Dynamic Retrieval Views Builder

Generates multiple semantic and structured retrieval views from ParsedQuery
deterministically without LLM calls.
"""

from typing import Dict, List
from app.models.parsed_query import ParsedQuery


def build_retrieval_views(parsed_query: ParsedQuery) -> Dict[str, str]:
    """
    Build dynamic retrieval views from a ParsedQuery.

    Returns a mapping of view_name -> view_query_text:
      - 'original': raw user query
      - 'semantic': intent-preserving natural language query
      - 'structured': compact, deterministic synthesis of concepts, relationships,
                      attributes, and requirements.
    """
    views: Dict[str, str] = {}

    # 1. View 1: Original query
    orig = (parsed_query.original_query or "").strip()
    if orig:
        views["original"] = orig

    # 2. View 2: Semantic query
    sem = (parsed_query.semantic_query or "").strip()
    if sem:
        views["semantic"] = sem
    elif orig:
        views["semantic"] = orig

    # 3. View 3: Structured semantic query
    structured_sections: List[str] = []

    # Concepts
    if parsed_query.concepts:
        clean_concepts = [c.strip() for c in parsed_query.concepts if c.strip()]
        if clean_concepts:
            structured_sections.append(f"Concepts:\n{', '.join(clean_concepts)}")

    # Relationships
    if parsed_query.relationships:
        rel_lines = []
        for r in parsed_query.relationships:
            subj = (r.subject or "").strip()
            rel = (r.relation or "").strip()
            obj = (r.object or "").strip()
            if subj and rel and obj:
                line = f"{subj} {rel} {obj}"
                if r.context and r.context.strip():
                    line += f" ({r.context.strip()})"
                rel_lines.append(line)
        if rel_lines:
            structured_sections.append(f"Relationships:\n" + "\n".join(rel_lines))

    # Attributes
    if parsed_query.attributes:
        attr_lines = []
        for a in parsed_query.attributes:
            c = (a.concept or "").strip()
            n = (a.name or "").strip()
            v = (a.value or "").strip()
            if c and n and v:
                attr_lines.append(f"{c} - {n}: {v}")
            elif c and v:
                attr_lines.append(f"{c}: {v}")
        if attr_lines:
            structured_sections.append(f"Attributes:\n" + "\n".join(attr_lines))

    # Requirements
    if parsed_query.requirements:
        req_lines = [req.strip() for req in parsed_query.requirements if req.strip()]
        if req_lines:
            structured_sections.append(f"Requirements:\n" + "\n".join(req_lines))

    if structured_sections:
        views["structured"] = "\n\n".join(structured_sections)
    elif sem:
        views["structured"] = sem
    elif orig:
        views["structured"] = orig

    return views

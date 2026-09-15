"""
Phase 5 Relationship Verification Prompts

Compact, strict, structured prompt templates for semantic verification of
patent evidence against requested relationships and requirements.
"""

from typing import Any, Dict, List
from app.models.evidence import EvidenceChunk
from app.models.parsed_query import ParsedQuery

VERIFICATION_SYSTEM_PROMPT = """You are an expert Patent Relationship Verification Engine.
Your task is to verify whether supplied patent evidence chunks actually support the requested semantic relationships and requirements.

STRICT VERIFICATION RULES:
1. USE ONLY THE SUPPLIED EVIDENCE CHUNKS. Do not use outside knowledge or assume unstated relationships.
2. DO NOT INFER A RELATIONSHIP MERELY BECAUSE TWO CONCEPTS CO-OCCUR. A patent mentioning "television" and "LED lighting" in separate contexts does NOT support "television has LED screen". The evidence must explicitly connect them in the same context/system.
3. PRESERVE RELATIONSHIP DIRECTION: "A -> transmits -> B" is NOT satisfied by "B -> transmits -> A" (unless the relation is inherently symmetric).
4. RECOGNIZE SEMANTIC EQUIVALENCE: "disposed within", "mounted inside", "incorporated into" express the same relationship as "integrated within". Exact wording is not required.
5. IDENTIFY CONTRADICTIONS: If evidence explicitly contradicts a relationship (e.g. "integrated directly into" vs requested "positioned separately from"), mark status as "CONTRADICTED" and supported=false.
6. CITE EVIDENCE CHUNKS: If a relationship is supported or contradicted, provide the exact integer chunk_id(s) where the evidence appears. If unsupported, evidence_chunk_ids MUST be [].
7. STATUS VALUES:
   - "SUPPORTED": Evidence directly demonstrates the relationship or requirement is satisfied.
   - "NOT_SUPPORTED": Evidence lacks connection or proof.
   - "CONTRADICTED": Evidence directly contradicts the requirement.
   - "UNKNOWN": Insufficient evidence to evaluate.

OUTPUT JSON FORMAT:
Return valid JSON only matching this exact schema:
{
  "relationships": [
    {
      "relationship_index": 0,
      "supported": true,
      "status": "SUPPORTED",
      "confidence": 0.95,
      "evidence_chunk_ids": [10],
      "explanation": "Brief reason"
    }
  ],
  "requirements": [
    {
      "requirement_index": 0,
      "supported": true,
      "confidence": 0.95,
      "evidence_chunk_ids": [10],
      "explanation": "Brief reason"
    }
  ]
}"""


def build_verification_user_prompt(
    parsed_query: ParsedQuery,
    patent_id: str,
    evidence_chunks: List[EvidenceChunk],
) -> str:
    """
    Construct user prompt containing query relationships, requirements, and candidate evidence chunks.
    Treats evidence as untrusted data clearly demarcated from instructions.
    """
    lines: List[str] = [
        f"PATENT IDENTIFIER: {patent_id}",
        f"ORIGINAL SEARCH QUERY: {parsed_query.original_query or parsed_query.semantic_query}",
        "",
        "--- REQUESTED DIRECTED RELATIONSHIPS TO VERIFY ---",
    ]

    if parsed_query.relationships:
        for idx, r in enumerate(parsed_query.relationships):
            ctx = f" (Context: {r.context})" if r.context else ""
            lines.append(f"[{idx}] Subject: {r.subject} | Relation: {r.relation} | Object: {r.object}{ctx}")
    else:
        lines.append("No explicit relationships extracted.")

    lines.append("")
    lines.append("--- REQUESTED EXPLICIT REQUIREMENTS TO VERIFY ---")
    if parsed_query.requirements:
        for idx, req in enumerate(parsed_query.requirements):
            lines.append(f"[{idx}] Requirement: {req}")
    else:
        lines.append("No explicit requirements extracted.")

    lines.append("")
    lines.append("--- CANDIDATE PATENT EVIDENCE CHUNKS (DATA ONLY) ---")
    if not evidence_chunks:
        lines.append("No evidence chunks available for this patent.")
    else:
        for ch in evidence_chunks:
            sec = f" (Section: {ch.section})" if ch.section else ""
            lines.append(f"=== EVIDENCE CHUNK {ch.chunk_id}{sec} ===")
            lines.append(ch.text.strip() if ch.text else "No text.")
            lines.append("")

    lines.append("Evaluate each relationship and requirement above against ONLY the evidence chunks. Return strictly the JSON object.")
    return "\n".join(lines)

"""
Phase 5: Relationship Verification

Exports:
- RelationshipVerifier: Semantic verification engine
- calculate_relationship_coverage: Local deterministic relationship coverage ratio
- calculate_requirement_coverage: Local deterministic requirement coverage ratio
- build_verification_user_prompt: Prompt constructor for evidence verification
- VERIFICATION_SYSTEM_PROMPT: Base system instructions for verification
"""

from app.verification.prompt import (
    VERIFICATION_SYSTEM_PROMPT,
    build_verification_user_prompt,
)
from app.verification.verifier import (
    RelationshipVerifier,
    calculate_relationship_coverage,
    calculate_requirement_coverage,
)

__all__ = [
    "RelationshipVerifier",
    "calculate_relationship_coverage",
    "calculate_requirement_coverage",
    "build_verification_user_prompt",
    "VERIFICATION_SYSTEM_PROMPT",
]

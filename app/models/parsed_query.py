"""
Parsed Query Models

Strongly-typed data structures representing the structured semantic
interpretation of a natural-language patent search query.
"""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class SemanticRelationship(BaseModel):
    """
    Explicit semantic relationship connecting technical concepts.
    Preserves relationship direction (subject -> relation -> object).
    """
    model_config = ConfigDict(frozen=True, extra="ignore")

    subject: str = Field(
        ...,
        description="Source entity or component in the relationship."
    )
    relation: str = Field(
        ...,
        description="Semantic predicate describing how subject connects to object (e.g. 'integrated into', 'illuminated by', 'having', 'attached to')."
    )
    object: str = Field(
        ...,
        description="Target entity or component in the relationship."
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context or modifier (e.g. 'automotive', 'structural', 'chemical')."
    )


class ConceptAttribute(BaseModel):
    """
    Attribute or property attached to a technical concept.
    """
    model_config = ConfigDict(frozen=True, extra="ignore")

    concept: str = Field(
        ...,
        description="Entity or concept this attribute describes."
    )
    name: str = Field(
        ...,
        description="Property or attribute name (e.g. 'flexibility', 'display technology', 'material')."
    )
    value: str = Field(
        ...,
        description="Attribute value (e.g. 'flexible', 'OLED', 'solid electrolyte')."
    )


class MetadataFilter(BaseModel):
    """
    Structured patent metadata constraint.
    """
    model_config = ConfigDict(frozen=True, extra="ignore")

    field: str = Field(
        ...,
        description="Canonical metadata field code (e.g. 'AS_EN', 'IN_EN', 'PY', 'AD', 'PNC', 'CPC', 'IPC') or resolved field name."
    )
    operator: str = Field(
        default="==",
        description="Comparison operator: '==', '!=', 'contains', 'in', '>', '>=', '<', '<=', 'between'."
    )
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(
        ...,
        description="Value or values to filter on."
    )
    raw_field: Optional[str] = Field(
        default=None,
        description="Original un-normalized field name extracted from natural language (e.g. 'assignee', 'filing date')."
    )


class ParsedQuery(BaseModel):
    """
    Complete structured semantic interpretation of a patent query.
    """
    model_config = ConfigDict(extra="ignore")

    original_query: str = Field(
        default="",
        description="Original raw query exactly as entered by the user."
    )
    semantic_query: str = Field(
        default="",
        description="Complete natural-language representation preserving full user intent for semantic search."
    )
    concepts: List[str] = Field(
        default_factory=list,
        description="Distinct technical concepts and entities extracted from the query."
    )
    relationships: List[SemanticRelationship] = Field(
        default_factory=list,
        description="Directed relationships between technical concepts."
    )
    attributes: List[ConceptAttribute] = Field(
        default_factory=list,
        description="Attributes and characteristics modifying specific concepts."
    )
    requirements: List[str] = Field(
        default_factory=list,
        description="Functional and technical requirements the patent must satisfy."
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Explicit numerical, physical, or operational constraints."
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Explicit negative requirements or excluded technologies."
    )
    metadata_filters: List[MetadataFilter] = Field(
        default_factory=list,
        description="Patent metadata constraints (assignee, inventor, date, year, country, classification codes)."
    )
    is_metadata_only: bool = Field(
        default=False,
        description="True ONLY when the query is exclusively about metadata constraints with zero semantic/technical content."
    )

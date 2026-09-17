"""
Dynamic Qdrant Metadata Filter Builder

Constructs Qdrant Filter objects from Phase 1 MetadataFilters using a centralized
field mapping and supporting dynamic comparison operators (==, !=, >, >=, <, <=, contains, in).
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

from app.models.parsed_query import MetadataFilter

# Centralized mapping from canonical codes to actual JSON metadata keys in Qdrant payloads
CANONICAL_TO_PAYLOAD_KEYS: Dict[str, List[str]] = {
    # Assignee / Applicant
    "AS_EN": ["Current Assignee Standardized", "Assignee/Applicant (Standardized) with Address", "Applicant First Organization"],
    "CAS_EN": ["Current Assignee Standardized", "Assignee/Applicant (Standardized) with Address"],
    "CA_EN": ["Current Assignee Standardized", "Assignee/Applicant (Standardized) with Address"],
    "AP_EN": ["Applicant First Organization", "Assignee/Applicant (Standardized) with Address"],
    "APS_EN": ["Assignee/Applicant (Standardized) with Address", "Applicant First Organization"],
    "AO_EN": ["Assignee/Applicant (Original) with Address"],
    # Inventor
    "IN_EN": ["Inventor", "Inventor First"],
    "INF_EN": ["Inventor First", "Inventor"],
    # Dates & Years
    "PY": ["Publication Year"],
    "PD": ["Publication Date", "Publication Year"],
    "AY": ["Application Year"],
    "AD": ["Application Date", "Application Year"],
    "PRY": ["Priority Year", "Earliest Priority Year"],
    "EPRY": ["Earliest Priority Year", "Priority Year"],
    # Country / Jurisdiction
    "PNC": ["Publication Country Code"],
    "AC": ["Application Country"],
    "PRC": ["Priority Country"],
    # Classification
    "CPC": ["CPC", "CPC Primary", "CPC - 12 Digit", "CPC - 4 Digit", "CPC - 8 Digit"],
    "CPCP": ["CPC Primary", "CPC"],
    "CPC12": ["CPC - 12 Digit"],
    "CPC4": ["CPC - 4 Digit"],
    "CPC8": ["CPC - 8 Digit"],
    "IPC": ["IPC", "IPC - 12 Digit", "IPC - 4 Digit", "IPC - 8 Digit"],
    "IPC12": ["IPC - 12 Digit"],
    "IPC4": ["IPC - 4 Digit"],
    "IPC8": ["IPC - 8 Digit"],
    # Status
    "LST": ["Legal Status (Filed/Granted/Ceased)"],
    "ALD": ["Legal State\n(Alive/Dead)"],
    "PT": ["Publication Type"],
    "APT": ["Applicant Type"],
    "AG_EN": ["Attorney/Agent"],
}


def _normalize_name_text(s: str) -> str:
    """
    Normalize a name/free-text value for comparison: lowercase and strip
    punctuation (commas, periods) so "Last, First" (the dataset's stored
    format for Inventor/Assignee) compares equal to a query written as
    "First Last", collapsing any resulting extra whitespace.
    """
    cleaned = re.sub(r"[,.]", " ", s.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def resolve_payload_field_names(field: str) -> List[str]:
    """
    Resolve a canonical field code or raw field name into valid Qdrant payload keys.
    """
    upper = field.strip().upper()
    if upper in CANONICAL_TO_PAYLOAD_KEYS:
        return CANONICAL_TO_PAYLOAD_KEYS[upper]

    # Deliberately NOT falling back to METADATA_FIELD_CODES[upper] here: that
    # table is a generic patent-field reference, and its description text
    # ("Attorney/Agent English") frequently doesn't match this dataset's
    # actual ingested payload key ("Attorney/Agent") - a wrong guess is
    # worse than no guess, since it causes a real, resolvable field to be
    # silently treated as absent and fail the whole filter. Only the
    # hand-verified CANONICAL_TO_PAYLOAD_KEYS table above is trusted.

    # Unknown field (e.g. a raw phrase the LLM invented like "cpc_12_digit"
    # or "attorney_or_agent" that never resolved to a canonical code) - we
    # have no reliable payload key to check, so return nothing rather than
    # guessing the literal string as a key. Guessing it almost never
    # matches the real Qdrant field name/casing, and callers treat "no
    # resolvable key" as "can't verify, don't reject the patent for it" -
    # very different from "resolved to a real key that this patent lacks".
    return []


def _is_name_field(payload_key: str) -> bool:
    """
    Fields holding free-text person/org names (inventor, assignee, etc.) or
    titles/addresses. These are stored with formatting (e.g. "Last, First")
    that varies from how a query states them, and there is no full-text
    payload index in this Qdrant collection to fuzzy-match them - so they
    must never get a hard Qdrant equality/MatchAny condition (that always
    requires an exact string match). Left to match_patent_metadata() for
    Python-side, punctuation-insensitive verification instead.
    """
    return any(kw in payload_key.lower() for kw in ["assignee", "inventor", "title", "address"])


def _build_condition_for_key(
    payload_key: str,
    operator: str,
    value: Any,
) -> Tuple[Optional[FieldCondition], bool]:
    """
    Build a single Qdrant FieldCondition.
    Returns (condition, is_must_not).
    """
    key_path = f'metadata."{payload_key}"'
    op = (operator or "==").strip().lower()
    is_name = _is_name_field(payload_key)

    if op in ("==", "eq"):
        if is_name:
            return None, False
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), False
        # Every field in this collection is stored as a string or list of
        # strings (confirmed against live payloads), even ones the
        # normalizer converts to int for range comparisons (e.g. year
        # fields: "2006" in Qdrant vs normalized int 2006). MatchValue is
        # type-sensitive, so an int here would never match the stored
        # string - always compare as string for exact equality.
        return FieldCondition(key=key_path, match=MatchValue(value=str(value))), False

    elif op in ("!=", "ne", "neq"):
        if is_name:
            return None, False
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), True
        return FieldCondition(key=key_path, match=MatchValue(value=str(value))), True

    elif op in ("contains", "like", "match", "includes", "has"):
        # No full-text index exists for MatchText to use on any field here,
        # so this always defers to match_patent_metadata() in Python.
        return None, False

    elif op in ("in", "any"):
        if is_name:
            return None, False
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), False
        return FieldCondition(key=key_path, match=MatchValue(value=str(value))), False

    elif op in (">", "gt"):
        num_val = _coerce_numeric(value)
        return FieldCondition(key=key_path, range=Range(gt=num_val)), False

    elif op in (">=", "gte"):
        num_val = _coerce_numeric(value)
        return FieldCondition(key=key_path, range=Range(gte=num_val)), False

    elif op in ("<", "lt"):
        num_val = _coerce_numeric(value)
        return FieldCondition(key=key_path, range=Range(lt=num_val)), False

    elif op in ("<=", "lte"):
        num_val = _coerce_numeric(value)
        return FieldCondition(key=key_path, range=Range(lte=num_val)), False

    elif op in ("between", "range"):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return FieldCondition(
                key=key_path,
                range=Range(
                    gte=_coerce_numeric(value[0]),
                    lte=_coerce_numeric(value[1]),
                ),
            ), False

    # Default fallback to MatchValue
    return FieldCondition(key=key_path, match=MatchValue(value=value)), False


def _coerce_numeric(val: Any) -> Union[int, float]:
    """Convert numeric string or number to float or int for Qdrant Range."""
    if isinstance(val, (int, float)):
        return val
    try:
        s = str(val).strip()
        if s.isdigit():
            return int(s)
        return float(s)
    except (ValueError, TypeError):
        return 0


def build_qdrant_filter(metadata_filters: List[MetadataFilter]) -> Optional[Filter]:
    """
    Convert a list of MetadataFilter instances into a unified Qdrant Filter.
    Supports single or multiple filters combined via 'must' and 'must_not'.
    """
    if not metadata_filters:
        return None

    must_conditions: List[FieldCondition] = []
    must_not_conditions: List[FieldCondition] = []

    for f in metadata_filters:
        if not f.field and not f.raw_field:
            continue

        field_name = f.field or f.raw_field or ""
        payload_keys = resolve_payload_field_names(field_name)

        if not payload_keys:
            continue

        # If multiple payload keys are possible (e.g. Current Assignee vs Applicant),
        # use the primary one for Qdrant index condition
        primary_key = payload_keys[0]
        cond, is_negated = _build_condition_for_key(primary_key, f.operator, f.value)

        if cond is not None:
            if is_negated:
                must_not_conditions.append(cond)
            else:
                must_conditions.append(cond)

    if not must_conditions and not must_not_conditions:
        return None

    return Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def match_patent_metadata(metadata: Dict[str, Any], metadata_filters: List[MetadataFilter]) -> bool:
    """
    Deterministic Python evaluation of metadata constraints against a patent's metadata dictionary.
    Used for verifying retrieved candidates and ensuring 100% precision across string and list fields.
    """
    if not metadata_filters:
        return True

    for f in metadata_filters:
        field_name = f.field or f.raw_field or ""
        payload_keys = resolve_payload_field_names(field_name)

        if not payload_keys:
            # Unknown field - we have no real payload key to check it
            # against, so we can't verify this constraint either way.
            # Don't reject an otherwise-matching patent just because one
            # filter refers to a field we don't know how to look up; that
            # would let a single unmapped field (out of possibly dozens on
            # a dense query) zero out every candidate.
            continue

        # Extract values present in metadata for any resolved key
        found_values: List[Any] = []
        for key in payload_keys:
            if key in metadata:
                val = metadata[key]
                if isinstance(val, list):
                    found_values.extend(val)
                else:
                    found_values.append(val)

        if not found_values:
            # Field resolved to a real payload key, but this patent
            # genuinely doesn't have it - a legitimate mismatch.
            if f.operator in ("!=", "ne", "neq"):
                continue
            return False

        target = f.value
        op = (f.operator or "==").strip().lower()

        matched = False
        for actual in found_values:
            actual_str = str(actual).strip()
            actual_norm = _normalize_name_text(actual_str)
            if op in ("==", "eq"):
                if isinstance(target, list):
                    if any(_normalize_name_text(str(t)) == actual_norm or _normalize_name_text(str(t)) in actual_norm for t in target):
                        matched = True
                        break
                elif _normalize_name_text(str(target)) == actual_norm or _normalize_name_text(str(target)) in actual_norm:
                    matched = True
                    break

            elif op in ("!=", "ne", "neq"):
                if _normalize_name_text(str(target)) == actual_norm:
                    matched = False
                    return False
                matched = True

            elif op in ("contains", "like", "match", "includes", "has"):
                if _normalize_name_text(str(target)) in actual_norm:
                    matched = True
                    break

            elif op in ("in", "any"):
                if isinstance(target, list):
                    if any(_normalize_name_text(str(t)) in actual_norm for t in target):
                        matched = True
                        break
                elif actual_norm in _normalize_name_text(str(target)):
                    matched = True
                    break

            elif op in (">", "gt"):
                try:
                    if float(actual_str) > float(target):
                        matched = True
                        break
                except ValueError:
                    if actual_str > str(target):
                        matched = True
                        break

            elif op in (">=", "gte"):
                try:
                    if float(actual_str) >= float(target):
                        matched = True
                        break
                except ValueError:
                    if actual_str >= str(target):
                        matched = True
                        break

            elif op in ("<", "lt"):
                try:
                    if float(actual_str) < float(target):
                        matched = True
                        break
                except ValueError:
                    if actual_str < str(target):
                        matched = True
                        break

            elif op in ("<=", "lte"):
                try:
                    if float(actual_str) <= float(target):
                        matched = True
                        break
                except ValueError:
                    if actual_str <= str(target):
                        matched = True
                        break

        if not matched:
            return False

    return True

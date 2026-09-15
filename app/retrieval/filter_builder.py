"""
Dynamic Qdrant Metadata Filter Builder

Constructs Qdrant Filter objects from Phase 1 MetadataFilters using a centralized
field mapping and supporting dynamic comparison operators (==, !=, >, >=, <, <=, contains, in).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    Range,
)

from app.models.parsed_query import MetadataFilter
from metadata_fields import METADATA_FIELD_CODES

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
}


def resolve_payload_field_names(field: str) -> List[str]:
    """
    Resolve a canonical field code or raw field name into valid Qdrant payload keys.
    """
    upper = field.strip().upper()
    if upper in CANONICAL_TO_PAYLOAD_KEYS:
        return CANONICAL_TO_PAYLOAD_KEYS[upper]

    # Check METADATA_FIELD_CODES description
    if upper in METADATA_FIELD_CODES:
        desc = METADATA_FIELD_CODES[upper]
        return [desc]

    # If it's already a full description/key
    return [field.strip()]


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

    if op in ("==", "eq"):
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), False
        if isinstance(value, str) and any(kw in payload_key.lower() for kw in ["assignee", "inventor", "title", "address"]):
            return FieldCondition(key=key_path, match=MatchText(text=str(value))), False
        return FieldCondition(key=key_path, match=MatchValue(value=value)), False

    elif op in ("!=", "ne", "neq"):
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), True
        return FieldCondition(key=key_path, match=MatchValue(value=value)), True

    elif op in ("contains", "like", "match", "includes", "has"):
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), False
        return FieldCondition(key=key_path, match=MatchText(text=str(value))), False

    elif op in ("in", "any"):
        if isinstance(value, list):
            return FieldCondition(key=key_path, match=MatchAny(any=[str(v) for v in value])), False
        return FieldCondition(key=key_path, match=MatchValue(value=value)), False

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
            # Field is missing from patent metadata
            if f.operator in ("!=", "ne", "neq"):
                continue
            return False

        target = f.value
        op = (f.operator or "==").strip().lower()

        matched = False
        for actual in found_values:
            actual_str = str(actual).strip()
            if op in ("==", "eq"):
                if isinstance(target, list):
                    if any(str(t).lower() == actual_str.lower() or str(t).lower() in actual_str.lower() for t in target):
                        matched = True
                        break
                elif str(target).lower() == actual_str.lower() or str(target).lower() in actual_str.lower():
                    matched = True
                    break

            elif op in ("!=", "ne", "neq"):
                if str(target).lower() == actual_str.lower():
                    matched = False
                    return False
                matched = True

            elif op in ("contains", "like", "match", "includes", "has"):
                if str(target).lower() in actual_str.lower():
                    matched = True
                    break

            elif op in ("in", "any"):
                if isinstance(target, list):
                    if any(str(t).lower() in actual_str.lower() for t in target):
                        matched = True
                        break
                elif actual_str.lower() in str(target).lower():
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

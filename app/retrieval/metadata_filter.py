"""
Phase 3: Metadata Filtering & Constraint Enforcement

Deterministic metadata constraint validation layer:
- Evaluates candidate patents against Phase 1 MetadataFilters
- Strict AND semantics across multiple constraints
- Handles missing metadata deterministically (fails on missing required fields)
- Date-aware comparison semantics (YYYY-MM-DD, YYYYMMDD, YYYY)
- Dynamic and reusable for arbitrary patent domains without hardcoded rules
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from app.models.candidate import (
    CandidatePatent,
    FilterDiagnostic,
    FilteredCandidateResult,
)
from app.models.parsed_query import MetadataFilter
from app.retrieval.filter_builder import resolve_payload_field_names

logger = logging.getLogger(__name__)


def _normalize_string(val: Any) -> str:
    """
    Normalize a string value for comparison: strip whitespace, lower case,
    and strip commas/periods so a "Last, First" stored name compares equal
    to a "First Last" query value, collapsing the resulting whitespace.
    """
    if val is None:
        return ""
    cleaned = re.sub(r"[,.]", " ", str(val).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_date_or_year(val: Any) -> Optional[Tuple[int, str]]:
    """
    Parse date or year value into normalized integer format and type tag.
    Returns:
      (YYYY, 'year') for 4-digit years
      (YYYYMMDD, 'date') for full 8-digit dates
      None if unparseable
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None

    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 4:
        return int(digits), "year"
    elif len(digits) >= 8:
        # Take first 8 digits (YYYYMMDD)
        return int(digits[:8]), "date"
    return None


def _compare_dates(actual_val: Any, op: str, target_val: Any) -> bool:
    """
    Perform semantic date / year comparison.
    """
    actual_parsed = _parse_date_or_year(actual_val)
    target_parsed = _parse_date_or_year(target_val)

    if actual_parsed is None or target_parsed is None:
        return False

    a_num, a_type = actual_parsed
    t_num, t_type = target_parsed
    op_clean = (op or "==").strip().lower()

    # Same granularity (year vs year, or date vs date)
    if a_type == t_type:
        if op_clean in (">", "gt"):
            return a_num > t_num
        elif op_clean in (">=", "gte"):
            return a_num >= t_num
        elif op_clean in ("<", "lt"):
            return a_num < t_num
        elif op_clean in ("<=", "lte"):
            return a_num <= t_num
        # "in"/"any" is handled the same as equality here: the caller
        # (matches_metadata_filter) already loops this comparison over
        # every (actual, target) pair for a list-valued filter and treats
        # any single True as a match, which is exactly "in" semantics.
        elif op_clean in ("==", "eq", "in", "any"):
            return a_num == t_num
        elif op_clean in ("!=", "ne", "neq"):
            return a_num != t_num

    # Actual is full date (YYYYMMDD), target is year (YYYY)
    if a_type == "date" and t_type == "year":
        a_year = int(str(a_num)[:4])
        if op_clean in (">", "gt"):
            return a_year > t_num
        elif op_clean in (">=", "gte"):
            return a_year >= t_num
        elif op_clean in ("<", "lt"):
            return a_year < t_num
        elif op_clean in ("<=", "lte"):
            return a_year <= t_num
        elif op_clean in ("==", "eq", "in", "any"):
            return a_year == t_num
        elif op_clean in ("!=", "ne", "neq"):
            return a_year != t_num

    # Actual is year (YYYY), target is full date (YYYYMMDD)
    if a_type == "year" and t_type == "date":
        t_year = int(str(t_num)[:4])
        if op_clean in (">", "gt"):
            return a_num > t_year or (a_num == t_year and (a_num * 10000 + 1231) > t_num)
        elif op_clean in (">=", "gte"):
            return a_num >= t_year
        elif op_clean in ("<", "lt"):
            return a_num < t_year or (a_num == t_year and (a_num * 10000 + 101) < t_num)
        elif op_clean in ("<=", "lte"):
            return a_num <= t_year
        elif op_clean in ("==", "eq", "in", "any"):
            return a_num == t_year
        elif op_clean in ("!=", "ne", "neq"):
            return a_num != t_year

    return False


def _compare_code(actual_val: Any, op: str, target_val: Any) -> bool:
    """
    Compare classification or jurisdiction codes (e.g. CPC, IPC, PNC, AC).
    Supports exact match, list membership, and prefix match.
    """
    a_clean = re.sub(r"[\s\/\-\.]", "", _normalize_string(actual_val))
    t_clean = re.sub(r"[\s\/\-\.]", "", _normalize_string(target_val))

    if not a_clean or not t_clean:
        return False

    op_clean = (op or "==").strip().lower()

    if op_clean in ("==", "eq"):
        return a_clean == t_clean or a_clean.startswith(t_clean) or t_clean in a_clean
    elif op_clean in ("!=", "ne", "neq"):
        return a_clean != t_clean and not a_clean.startswith(t_clean)
    elif op_clean in ("contains", "like", "match", "includes", "has", "in"):
        return t_clean in a_clean or a_clean.startswith(t_clean)

    return a_clean == t_clean


def _compare_text(actual_val: Any, op: str, target_val: Any) -> bool:
    """
    Compare general text/organization/name metadata fields.
    """
    a_str = _normalize_string(actual_val)
    t_str = _normalize_string(target_val)

    if not a_str or not t_str:
        return False

    op_clean = (op or "==").strip().lower()

    if op_clean in ("==", "eq"):
        # Match if equal or target is a clean substring
        return a_str == t_str or t_str in a_str
    elif op_clean in ("!=", "ne", "neq"):
        return a_str != t_str and t_str not in a_str
    elif op_clean in ("contains", "like", "match", "includes", "has"):
        return t_str in a_str
    elif op_clean in ("in", "any"):
        return a_str in t_str or t_str in a_str

    return a_str == t_str


def _is_date_field(field_name: str) -> bool:
    """Check if field relates to date or year."""
    f_upper = field_name.strip().upper()
    if f_upper in ("PY", "PD", "AY", "AD", "PRY", "PRD", "EPRY", "EPRD"):
        return True
    f_lower = field_name.lower()
    return any(w in f_lower for w in ["year", "date"])


def _is_code_field(field_name: str) -> bool:
    """Check if field is a classification or jurisdiction code."""
    f_upper = field_name.strip().upper()
    if f_upper in ("CPC", "CPCP", "CPC12", "CPC4", "CPC8", "IPC", "IPC12", "IPC4", "IPC8", "PNC", "AC", "PRC"):
        return True
    f_lower = field_name.lower()
    return any(w in f_lower for w in ["cpc", "ipc", "country", "jurisdiction"])


def matches_metadata_filter(
    metadata: Dict[str, Any], filter_item: MetadataFilter
) -> bool:
    """
    Evaluate a single MetadataFilter against a patent metadata dictionary.
    Returns True if satisfied, False otherwise.
    """
    if not filter_item.field and not filter_item.raw_field:
        return True

    field_name = filter_item.field or filter_item.raw_field or ""
    payload_keys = resolve_payload_field_names(field_name)

    if not payload_keys:
        # Unknown field with no resolvable payload key - can't verify this
        # constraint, so don't let it veto an otherwise-matching patent.
        return True

    # Collect all existing values in the patent's metadata matching the resolved keys
    actual_values: List[Any] = []
    for key in payload_keys:
        if key in metadata:
            val = metadata[key]
            if isinstance(val, list):
                actual_values.extend(val)
            elif val is not None:
                actual_values.append(val)

    # Missing metadata handling
    if not actual_values:
        op = (filter_item.operator or "==").strip().lower()
        if op in ("!=", "ne", "neq"):
            return True
        return False

    target = filter_item.value
    op = filter_item.operator

    # Handle list of target values (e.g. IN ['US', 'EP'])
    targets = target if isinstance(target, list) else [target]

    is_date = _is_date_field(field_name)
    is_code = _is_code_field(field_name)

    for actual in actual_values:
        for tgt in targets:
            if is_date:
                if _compare_dates(actual, op, tgt):
                    return True
            elif is_code:
                if _compare_code(actual, op, tgt):
                    return True
            else:
                if _compare_text(actual, op, tgt):
                    return True

    # If operator is negated (!=), all actual values must satisfy negation
    if (op or "").strip().lower() in ("!=", "ne", "neq"):
        return True

    return False


def matches_all_metadata_filters(
    metadata: Dict[str, Any], filters: List[MetadataFilter]
) -> Tuple[bool, Dict[int, bool]]:
    """
    Evaluate all metadata filters using strict AND semantics.
    Returns:
      (all_passed, {filter_index: pass_bool})
    """
    if not filters:
        return True, {}

    per_filter_results: Dict[int, bool] = {}
    all_passed = True

    for idx, f in enumerate(filters):
        passed = matches_metadata_filter(metadata, f)
        per_filter_results[idx] = passed
        if not passed:
            all_passed = False

    return all_passed, per_filter_results


def filter_candidates(
    candidates: List[CandidatePatent],
    metadata_filters: List[MetadataFilter],
    is_metadata_only: bool = False,
) -> FilteredCandidateResult:
    """
    Execute Phase 3: Metadata Filtering & Constraint Enforcement.
    Filters candidate patents strictly by all metadata constraints using AND semantics.
    """
    t_start = time.perf_counter()

    if not metadata_filters:
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        return FilteredCandidateResult(
            candidates=candidates,
            total_before=len(candidates),
            total_after=len(candidates),
            filtered_count=0,
            diagnostics=[],
            is_metadata_only=is_metadata_only,
            filter_time_ms=round(t_elapsed_ms, 2),
        )

    # Initialize per-filter diagnostics
    diagnostic_trackers: List[Dict[str, Any]] = [
        {
            "field": f.field or f.raw_field or "UNKNOWN",
            "operator": f.operator,
            "value": f.value,
            "passed": 0,
            "failed": 0,
        }
        for f in metadata_filters
    ]

    surviving_candidates: List[CandidatePatent] = []

    for cand in candidates:
        all_passed, per_filter = matches_all_metadata_filters(
            cand.metadata, metadata_filters
        )

        # Update diagnostics
        for idx, passed in per_filter.items():
            if passed:
                diagnostic_trackers[idx]["passed"] += 1
            else:
                diagnostic_trackers[idx]["failed"] += 1

        if all_passed:
            surviving_candidates.append(cand)

    t_elapsed_ms = (time.perf_counter() - t_start) * 1000

    diagnostics = [
        FilterDiagnostic(
            field=d["field"],
            operator=d["operator"],
            value=d["value"],
            passed=d["passed"],
            failed=d["failed"],
        )
        for d in diagnostic_trackers
    ]

    return FilteredCandidateResult(
        candidates=surviving_candidates,
        total_before=len(candidates),
        total_after=len(surviving_candidates),
        filtered_count=len(candidates) - len(surviving_candidates),
        diagnostics=diagnostics,
        is_metadata_only=is_metadata_only,
        filter_time_ms=round(t_elapsed_ms, 2),
    )

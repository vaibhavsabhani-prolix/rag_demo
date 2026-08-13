"""
Filter Engine

Applies structured MetadataFilter constraints (app/query_understanding/models.py)
against patent metadata AFTER vector search (post-retrieval filtering).

The pipeline is:
    1. Pure Qdrant vector search, grouped by patent_id → top PATENT_CANDIDATE_TOP_K patents
    2. Extract unique patent_ids from those candidates
    3. Batch fetch metadata for only those patent_ids
    4. FilterEngine.matches() → keep only chunks belonging to matching patents
    5. Reranker scores every surviving chunk → aggregate by patent → top FINAL_TOP_K patents

FilterEngine validates every filter against FIELD_MAPPING before
evaluating it:
    - Field must exist in FIELD_MAPPING
    - Operator must be in the field's allowed operators list

Python-side predicate (matches): evaluates each filter against the
patent's stored metadata dict (using the real_key from FIELD_MAPPING
to look up the actual stored value).

Native Qdrant Filter support (to_qdrant_filter, split_native_and_python):
retained for future use when Qdrant-compatible fields/operators exist.
Currently all fields have keys with spaces/special characters that are
unsafe for Qdrant's dot-notation paths (see QDRANT_UNSAFE_FIELDS).
"""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from app.query_understanding.field_mapping import FIELD_MAPPING, QDRANT_UNSAFE_FIELDS
from app.query_understanding.models import MetadataFilter

# Operators with a clean, non-substring native Qdrant translation.
# "contains" is deliberately excluded - see module docstring.
NATIVE_OPERATORS = {"equals", "gt", "gte", "lt", "lte"}


class UnsupportedFilterFieldError(ValueError):
    """Raised when a MetadataFilter references a field outside FIELD_MAPPING."""


class UnsupportedOperatorError(ValueError):
    """Raised when a MetadataFilter's operator doesn't apply to its field's type."""


def _check_filter(f: MetadataFilter) -> dict:
    """Validate field existence and operator/type compatibility; return the field's spec."""

    spec = FIELD_MAPPING.get(f.field)
    if spec is None:
        raise UnsupportedFilterFieldError(
            f"'{f.field}' is not in the filterable metadata allowlist (FIELD_MAPPING)."
        )

    if f.operator not in spec["operators"]:
        raise UnsupportedOperatorError(
            f"Operator '{f.operator}' is not valid for '{f.field}' "
            f"(type={spec['type']!r}, supported operators={spec['operators']})."
        )

    return spec


class FilterEngine:

    # ==============================================================
    # Routing: which filters can use a native Qdrant Filter
    # ==============================================================

    @staticmethod
    def split_native_and_python(
        filters: list[MetadataFilter],
    ) -> tuple[list[MetadataFilter], list[MetadataFilter]]:
        """
        Partition *filters* into (native_ok, python_only). Every filter
        is validated (field + operator) here regardless of which bucket
        it ends up in.
        """

        native, python_only = [], []

        for f in filters:
            _check_filter(f)
            if f.operator in NATIVE_OPERATORS and f.field not in QDRANT_UNSAFE_FIELDS:
                native.append(f)
            else:
                python_only.append(f)

        return native, python_only

    # ==============================================================
    # Native Qdrant filter
    # ==============================================================

    @staticmethod
    def to_qdrant_filter(filters: list[MetadataFilter]) -> Filter | None:
        """
        Build a native Qdrant Filter from *filters* (all must already be
        native-eligible per split_native_and_python). Returns None for
        an empty list, meaning "no server-side narrowing" rather than
        "match nothing".
        """

        if not filters:
            return None

        conditions = []

        for f in filters:
            spec = _check_filter(f)
            key = f"metadata.{spec['real_key']}"

            if f.operator == "equals":
                conditions.append(FieldCondition(key=key, match=MatchValue(value=f.value)))
            elif f.operator == "gt":
                conditions.append(FieldCondition(key=key, range=Range(gt=f.value)))
            elif f.operator == "gte":
                conditions.append(FieldCondition(key=key, range=Range(gte=f.value)))
            elif f.operator == "lt":
                conditions.append(FieldCondition(key=key, range=Range(lt=f.value)))
            elif f.operator == "lte":
                conditions.append(FieldCondition(key=key, range=Range(lte=f.value)))

        return Filter(must=conditions)

    # ==============================================================
    # Python-side predicate
    # ==============================================================

    @staticmethod
    def matches(metadata: dict, filters: list[MetadataFilter]) -> bool:
        return all(FilterEngine._matches_one(metadata, f) for f in filters)

    @staticmethod
    def _matches_one(metadata: dict, f: MetadataFilter) -> bool:
        """
        A patent with no stored value for the field never matches -
        including a "not_equals"/"not_contains" exclusion, since there's
        nothing on record to confirm the exclusion against.
        """

        spec = _check_filter(f)

        actual = metadata.get(spec["real_key"])
        if actual is None:
            return False

        op = f.operator

        if isinstance(actual, list):
            values = [str(v) for v in actual]

            if op in ("contains", "not_contains"):
                found = any(str(f.value).lower() in v.lower() for v in values)
                return not found if op == "not_contains" else found

            try:
                numbers = [float(v) for v in values]
            except ValueError:
                return False
            return any(FilterEngine._compare(n, op, float(f.value)) for n in numbers)

        if op in ("contains", "not_contains"):
            found = str(f.value).lower() in str(actual).lower()
            return not found if op == "not_contains" else found

        if op in ("equals", "not_equals"):
            try:
                equal = float(actual) == float(f.value)
            except (TypeError, ValueError):
                equal = str(actual).lower() == str(f.value).lower()
            return not equal if op == "not_equals" else equal

        try:
            return FilterEngine._compare(float(actual), op, float(f.value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _compare(actual: float, op: str, target: float) -> bool:
        if op == "gt":
            return actual > target
        if op == "gte":
            return actual >= target
        if op == "lt":
            return actual < target
        if op == "lte":
            return actual <= target
        return False

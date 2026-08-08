"""
Test cases for LLM-based Query Understanding & FIELD_MAPPING resolution.

Run with:
    ./.venv/bin/python -m app.test_query_understanding
"""

import sys

from app.query_understanding import QueryUnderstanding
from app.query_understanding.field_mapping import FIELD_MAPPING
from app.query_understanding.parser import resolve_filter


def case_1_direct_code_resolves(qu):
    """The LLM's field code maps straight to the right FIELD_MAPPING entry."""
    f = resolve_filter("PY", "equals", "2008")
    assert f is not None
    assert f.field == "publication_year"
    assert f.operator == "equals"
    assert f.value == 2008


def case_2_comparator_operator_preserved(qu):
    """A valid comparator operator for a numeric field is kept as-is."""
    f = resolve_filter("PY", "gt", "2018")
    assert f is not None
    assert f.field == "publication_year"
    assert f.operator == "gt"
    assert f.value == 2018


def case_3_unsupported_operator_falls_back(qu):
    """An operator not valid for the field's type falls back to the field's first allowed operator."""
    f = resolve_filter("PY", "contains", "2018")
    assert f is not None
    assert f.operator in FIELD_MAPPING["publication_year"]["operators"]
    assert f.operator == "equals"


def case_4_unknown_code_not_invented(qu):
    """An unknown/invented field code is dropped, never invents a field."""
    assert resolve_filter("BOTTLE_COLOR", "equals", "red") is None
    assert resolve_filter("XYZ", "equals", "red") is None


def case_5_country_code_normalized(qu):
    """Country field values are normalized to ISO-ish codes."""
    f = resolve_filter("AC", "equals", "United States")
    assert f is not None
    assert f.field == "application_country"
    assert f.value == "US"

    f2 = resolve_filter("PNC", "equals", "Japan")
    assert f2 is not None
    assert f2.field == "publication_country_code"
    assert f2.value == "JP"


def case_6_unrecognized_country_dropped(qu):
    """A country value that doesn't resolve to a known code is dropped."""
    assert resolve_filter("AC", "equals", "Nowhereland") is None


def case_7_org_name_normalized(qu):
    """Organization field values are uppercased/cleaned."""
    f = resolve_filter("CAN_EN", "contains", "coca-cola")
    assert f is not None
    assert f.field == "current_assignee_normalized"
    assert f.value == "COCA COLA"


def case_8_enum_value_matched_case_insensitively(qu):
    """Enum fields match against their declared values, case-insensitively."""
    f = resolve_filter("LST", "equals", "granted")
    assert f is not None
    assert f.field == "legal_status"
    assert f.value == "Granted"

    assert resolve_filter("LST", "equals", "not-a-real-status") is None


def case_9_empty_value_dropped(qu):
    """A blank value never produces a filter."""
    assert resolve_filter("PY", "equals", "") is None
    assert resolve_filter("PY", "equals", None) is None


def case_10_all_resolved_fields_in_field_mapping(qu):
    """Every resolved filter's field must be a key in FIELD_MAPPING."""
    cases = [
        ("PY", "equals", "2008"),
        ("AC", "equals", "US"),
        ("CAN_EN", "contains", "Coca Cola"),
        ("LST", "equals", "Granted"),
    ]
    for code, op, val in cases:
        f = resolve_filter(code, op, val)
        assert f is not None
        assert f.field in FIELD_MAPPING


def case_11_llm_unavailable_falls_back_to_pure_semantic(qu):
    """If the LLM is unavailable, the query passes through unfiltered."""
    qu_no_llm = QueryUnderstanding(use_llm=False)
    parsed = qu_no_llm.parse("bottle designs patented by Coca Cola in the US")

    assert parsed.semantic_query == "bottle designs patented by Coca Cola in the US"
    assert parsed.metadata_filters == []


def case_12_empty_query(qu):
    """An empty query is returned as-is, with no filters."""
    qu_no_llm = QueryUnderstanding(use_llm=False)
    parsed = qu_no_llm.parse("   ")
    assert parsed.metadata_filters == []


CASES = [
    ("Test 1  - direct code resolves to field", case_1_direct_code_resolves),
    ("Test 2  - comparator operator preserved", case_2_comparator_operator_preserved),
    ("Test 3  - unsupported operator falls back", case_3_unsupported_operator_falls_back),
    ("Test 4  - unknown code not invented", case_4_unknown_code_not_invented),
    ("Test 5  - country code normalized", case_5_country_code_normalized),
    ("Test 6  - unrecognized country dropped", case_6_unrecognized_country_dropped),
    ("Test 7  - org name normalized", case_7_org_name_normalized),
    ("Test 8  - enum value matched case-insensitively", case_8_enum_value_matched_case_insensitively),
    ("Test 9  - empty value dropped", case_9_empty_value_dropped),
    ("Test 10 - all resolved fields in FIELD_MAPPING", case_10_all_resolved_fields_in_field_mapping),
    ("Test 11 - LLM unavailable falls back to pure semantic", case_11_llm_unavailable_falls_back_to_pure_semantic),
    ("Test 12 - empty query", case_12_empty_query),
]


def main():
    qu = QueryUnderstanding(use_llm=False)
    failures = 0

    for name, case in CASES:
        try:
            case(qu)
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {name}: {exc!r}")
        else:
            print(f"[PASS] {name}")

    print()
    if failures:
        print(f"{failures}/{len(CASES)} case(s) failed.")
        sys.exit(1)

    print(f"All {len(CASES)} cases passed.")


if __name__ == "__main__":
    main()

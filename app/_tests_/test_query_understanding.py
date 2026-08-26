"""
Test cases for LLM-based Query Understanding & FIELD_MAPPING resolution.

Run with:
    ./.venv/bin/python -m app._tests_.test_query_understanding
"""

import sys

from app.filter_engine import FilterEngine
from app.query_understanding import QueryUnderstanding
from app.query_understanding.field_mapping import FIELD_MAPPING
from app.query_understanding.parser import group_duplicate_value_filters, resolve_filter


def case_1_direct_code_resolves(qu):
    """The LLM's field code maps straight to the right FIELD_MAPPING entry."""
    f = resolve_filter("PY", "equals", "2008")
    assert f is not None
    assert f.field == "publication_year"
    assert f.operator == "equals"
    assert f.value == 2008


def case_1b_equals_matches_array_valued_field(qu):
    """
    Regression: an array_number field (e.g. PRY - a patent can have more
    than one priority year) stored as a list must still match "equals"/
    "not_equals" against any element in the list, not just gt/gte/lt/lte -
    FilterEngine._matches_one()'s list branch previously fell through to
    _compare(), which only implements the four comparator operators and
    silently returned False for equals/not_equals, so a value genuinely
    present in the list never matched.
    """
    f = resolve_filter("PRY", "equals", "2012")
    assert f is not None
    assert FilterEngine.matches({"Priority Year": ["2012", "2013"]}, [f])

    not_eq = resolve_filter("PRY", "not_equals", "2012")
    assert not_eq is not None
    assert not FilterEngine.matches({"Priority Year": ["2012", "2013"]}, [not_eq])
    assert FilterEngine.matches({"Priority Year": ["2013", "2014"]}, [not_eq])


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


def case_7b_org_name_strips_legal_suffix_to_match_standardized_storage(qu):
    """
    Stored "standardized" assignee/applicant values already have the legal
    suffix stripped (e.g. "ALIOS BIOPHARMA", not "... INC") - the query
    value must be normalized the same way, or a verbatim "... INC" value
    (correct per RULE 3) would never match via FilterEngine's substring
    "contains" check.
    """
    f = resolve_filter("AAPS", "contains", "ALIOS BIOPHARMA INC")
    assert f is not None
    assert f.value == "ALIOS BIOPHARMA"

    assert FilterEngine.matches(
        {"Assignee/Applicant (Standardized) with Address": ["ALIOS BIOPHARMA"]}, [f]
    )


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


def case_13_same_value_on_different_fields_is_grouped(qu):
    """
    The LLM lists the same (operator, value) against several plausible real
    field codes (e.g. unsure whether "AP" is Application/Publication/
    Priority/Assignee country) - group_duplicate_value_filters() detects
    that pattern after resolve_filter() and tags them with a shared group.
    """
    candidates = [
        resolve_filter("AC", "equals", "AP"),
        resolve_filter("PNC", "equals", "AP"),
        resolve_filter("PRC", "equals", "AP"),
    ]
    assert all(candidates)
    grouped = group_duplicate_value_filters(candidates)

    groups = {f.group for f in grouped}
    assert len(groups) == 1 and None not in groups
    assert {f.field for f in grouped} == {
        "application_country",
        "publication_country_code",
        "priority_country",
    }


def case_14_single_field_value_stays_ungrouped(qu):
    """A value that only ever resolved to ONE field is untouched (group stays None)."""
    only_one = resolve_filter("PY", "equals", "2008")
    assert only_one is not None
    grouped = group_duplicate_value_filters([only_one])
    assert grouped[0].group is None


def case_15_distinct_values_never_grouped(qu):
    """Two genuinely different (field, value) constraints are never grouped - still strict AND."""
    china = resolve_filter("PRC", "equals", "China")
    filed = resolve_filter("LST", "equals", "Filed")
    assert china is not None and filed is not None
    grouped = group_duplicate_value_filters([china, filed])
    assert all(f.group is None for f in grouped)


def case_16_grouped_filters_or_within_group_and_and_alongside_ungrouped(qu):
    """FilterEngine.matches() ORs same-group filters, ANDs a group against an ungrouped filter."""
    country_candidates = [
        resolve_filter("AC", "equals", "AP"),
        resolve_filter("PRC", "equals", "AP"),
    ]
    assert all(country_candidates)
    filed = resolve_filter("LST", "equals", "Filed")
    assert filed is not None

    filters = group_duplicate_value_filters(country_candidates) + [filed]

    # Country OR-group satisfied via AC, ungrouped LST also satisfied.
    assert FilterEngine.matches(
        {
            "Application Country": "AP",
            "Legal Status (Filed/Granted/Ceased)": "Filed",
        },
        filters,
    )
    # Country OR-group satisfied via a *different* field (PRC) than the first case.
    assert FilterEngine.matches(
        {
            "Priority Country": "AP",
            "Legal Status (Filed/Granted/Ceased)": "Filed",
        },
        filters,
    )
    # Country OR-group unsatisfied (neither field holds "AP") even though LST matches.
    assert not FilterEngine.matches(
        {
            "Application Country": "US",
            "Legal Status (Filed/Granted/Ceased)": "Filed",
        },
        filters,
    )
    # Country OR-group satisfied but the ungrouped LST filter is not.
    assert not FilterEngine.matches(
        {
            "Application Country": "AP",
            "Legal Status (Filed/Granted/Ceased)": "Granted",
        },
        filters,
    )


def case_17_group_filter_never_routed_to_native_qdrant_path(qu):
    """A grouped filter is always routed python-only, never native (would silently AND, not OR)."""
    candidates = [resolve_filter("AC", "equals", "AP"), resolve_filter("PRC", "equals", "AP")]
    assert all(candidates)
    group = group_duplicate_value_filters(candidates)

    native, python_only = FilterEngine.split_native_and_python(group)
    assert native == []
    assert len(python_only) == len(group)


def case_17b_current_assignee_variants_and_combined_field_or_together(qu):
    """
    Regression: an assignee's "Current Assignee Normalized" can be empty
    while "Current Assignee Standardized" (a different field - just a
    different text-cleaning variant of the same current-owner fact) is
    populated. "assigned to X" must list every Current Assignee variant
    plus the combined Assignee/Applicant field as an OR group, or a patent
    populated under only one of them is wrongly missed.
    """
    candidates = [
        resolve_filter("CAN_EN", "contains", "ALIOS BIOPHARMA"),
        resolve_filter("CAS_EN", "contains", "ALIOS BIOPHARMA"),
        resolve_filter("AAPS", "contains", "ALIOS BIOPHARMA"),
    ]
    assert all(candidates)
    filters = group_duplicate_value_filters(candidates)

    assert len({f.group for f in filters}) == 1

    # Normalized empty, Standardized populated - exactly the reported case.
    assert FilterEngine.matches(
        {
            "Current Assignee Normalized": [],
            "Current Assignee Standardized": ["ALIOS BIOPHARMA"],
        },
        filters,
    )
    # Neither Current Assignee variant populated, only the combined field.
    assert FilterEngine.matches(
        {"Assignee/Applicant (Standardized) with Address": ["ALIOS BIOPHARMA"]},
        filters,
    )
    # None of the three populated - correctly no match.
    assert not FilterEngine.matches({}, filters)


def case_18_parse_groups_duplicate_llm_filter_entries(qu):
    """End-to-end: parse() groups filter entries the LLM explicitly marked "uncertain": true."""
    from unittest.mock import patch

    fake_response = {
        "semantic_query": None,
        "filters": [
            {"field": "AC", "operator": "equals", "value": "AP", "uncertain": True},
            {"field": "PNC", "operator": "equals", "value": "AP", "uncertain": True},
            {"field": "PRC", "operator": "equals", "value": "AP", "uncertain": True},
            {"field": "LST", "operator": "equals", "value": "Filed"},
        ],
        "intent": "patents from AP that are filed",
        "is_question": False,
    }

    qu_llm = QueryUnderstanding(use_llm=False)
    qu_llm.use_llm = True
    qu_llm._remote_available = True

    with patch.object(qu_llm, "_call_remote_llm", return_value=fake_response):
        parsed = qu_llm.parse("Patents from AP that are filed")

    groups = {f.group for f in parsed.metadata_filters}
    assert None in groups  # the LST filter stays ungrouped
    assert len(groups) == 2  # one shared group id + None
    grouped_fields = {f.field for f in parsed.metadata_filters if f.group is not None}
    assert grouped_fields == {
        "application_country",
        "publication_country_code",
        "priority_country",
    }


def case_19_explicit_coincidental_value_not_grouped(qu):
    """
    Regression: "application country AP" and "publication country AP"
    stated explicitly and separately in the same query must NOT be
    OR-grouped just because they share the value "AP" - the LLM was
    confident about both (so neither is marked "uncertain": true), and a
    coincidental value match must never turn two deliberate, independent
    constraints into alternatives to each other.
    """
    from unittest.mock import patch

    fake_response = {
        "semantic_query": None,
        "filters": [
            {"field": "AC", "operator": "equals", "value": "AP"},
            {"field": "PNC", "operator": "equals", "value": "AP"},
        ],
        "intent": "application country AP and publication country AP",
        "is_question": False,
    }

    qu_llm = QueryUnderstanding(use_llm=False)
    qu_llm.use_llm = True
    qu_llm._remote_available = True

    with patch.object(qu_llm, "_call_remote_llm", return_value=fake_response):
        parsed = qu_llm.parse("application country AP, publication country AP")

    assert len(parsed.metadata_filters) == 2
    assert all(f.group is None for f in parsed.metadata_filters)

    # Both independently required (AND): a patent with only one of the two
    # must NOT match.
    assert not FilterEngine.matches(
        {"Application Country": "AP", "Publication Country Code": "US"},
        parsed.metadata_filters,
    )
    assert FilterEngine.matches(
        {"Application Country": "AP", "Publication Country Code": "AP"},
        parsed.metadata_filters,
    )


CASES = [
    ("Test 1  - direct code resolves to field", case_1_direct_code_resolves),
    ("Test 1b - equals matches array-valued field", case_1b_equals_matches_array_valued_field),
    ("Test 2  - comparator operator preserved", case_2_comparator_operator_preserved),
    ("Test 3  - unsupported operator falls back", case_3_unsupported_operator_falls_back),
    ("Test 4  - unknown code not invented", case_4_unknown_code_not_invented),
    ("Test 5  - country code normalized", case_5_country_code_normalized),
    ("Test 6  - unrecognized country dropped", case_6_unrecognized_country_dropped),
    ("Test 7  - org name normalized", case_7_org_name_normalized),
    ("Test 7b - org name strips legal suffix to match standardized storage", case_7b_org_name_strips_legal_suffix_to_match_standardized_storage),
    ("Test 8  - enum value matched case-insensitively", case_8_enum_value_matched_case_insensitively),
    ("Test 9  - empty value dropped", case_9_empty_value_dropped),
    ("Test 10 - all resolved fields in FIELD_MAPPING", case_10_all_resolved_fields_in_field_mapping),
    ("Test 11 - LLM unavailable falls back to pure semantic", case_11_llm_unavailable_falls_back_to_pure_semantic),
    ("Test 12 - empty query", case_12_empty_query),
    ("Test 13 - same value on different fields is grouped", case_13_same_value_on_different_fields_is_grouped),
    ("Test 14 - single-field value stays ungrouped", case_14_single_field_value_stays_ungrouped),
    ("Test 15 - distinct values never grouped", case_15_distinct_values_never_grouped),
    ("Test 16 - grouped filters OR within group, AND alongside ungrouped", case_16_grouped_filters_or_within_group_and_and_alongside_ungrouped),
    ("Test 17 - group filter never routed to native Qdrant path", case_17_group_filter_never_routed_to_native_qdrant_path),
    ("Test 17b - Current Assignee variants + combined field OR together", case_17b_current_assignee_variants_and_combined_field_or_together),
    ("Test 18 - parse() groups duplicate LLM filter entries end-to-end", case_18_parse_groups_duplicate_llm_filter_entries),
    ("Test 19 - explicit coincidental value not grouped", case_19_explicit_coincidental_value_not_grouped),
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

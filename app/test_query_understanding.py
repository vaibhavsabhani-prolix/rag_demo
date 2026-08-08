"""
Test cases for LLM-based Query Understanding & FIELD_MAPPING resolution.

Run with:
    ./.venv/bin/python -m app.test_query_understanding
"""

import sys

from app.query_understanding import QueryUnderstanding
from app.query_understanding.field_mapping import FIELD_MAPPING
from app.query_understanding.parser import resolve_candidate_filter


def _filter(parsed, field):
    return next((f for f in parsed.metadata_filters if f.field == field), None)


def case_1_pure_semantic(qu):
    """Pure semantic query with no metadata filters."""
    parsed = qu.parse("bottle design")
    assert parsed.semantic_query in ("bottle design", "bottle designs")
    assert parsed.metadata_filters == []


def case_2_assignee_filter(qu):
    """Assignee filter: 'assigned to Coca Cola' -> current_assignee_normalized."""
    parsed = qu.parse("bottle design patents assigned to Coca Cola")
    assert "bottle design" in parsed.semantic_query
    assert len(parsed.metadata_filters) >= 1
    f = _filter(parsed, "current_assignee_normalized")
    assert f is not None
    assert f.operator == "contains"
    assert f.value == "COCA COLA"


def case_3_country_trailing_idiom(qu):
    """Country filter: 'patents in the US' -> application_country."""
    parsed = qu.parse("bottle design patents in the US")
    assert "bottle design" in parsed.semantic_query
    assert len(parsed.metadata_filters) >= 1
    f = _filter(parsed, "application_country")
    assert f is not None
    assert f.operator == "equals"
    assert f.value == "US"


def case_4_country_leading_idiom(qu):
    """Country before topic: 'US bottle design patents' -> application_country."""
    parsed = qu.parse("US bottle design patents")
    assert "bottle design" in parsed.semantic_query
    assert len(parsed.metadata_filters) >= 1
    f = _filter(parsed, "application_country")
    assert f is not None
    assert f.operator == "equals"
    assert f.value == "US"


def case_5_application_year(qu):
    """Application year: 'filed in 2020' -> application_year equals 2020."""
    parsed = qu.parse("bottle design patents filed in 2020")
    assert "bottle design" in parsed.semantic_query
    assert len(parsed.metadata_filters) >= 1
    f = _filter(parsed, "application_year")
    assert f is not None
    assert f.operator == "equals"
    assert f.value == 2020


def case_6_country_plus_year_gt(qu):
    """Publication country + year: 'published in Japan after 2018'."""
    parsed = qu.parse("bottle design patents published in Japan after 2018")
    assert "bottle design" in parsed.semantic_query

    country = _filter(parsed, "publication_country_code")
    assert country is not None and country.operator == "equals" and country.value == "JP"

    year = _filter(parsed, "publication_year")
    assert year is not None and year.operator == "gt" and year.value == 2018


def case_7_bare_assignee_plus_enum(qu):
    """Legal state + assignee: 'active Coca Cola bottle patents'."""
    parsed = qu.parse("active Coca Cola bottle patents")
    assert "bottle" in parsed.semantic_query

    assignee = _filter(parsed, "current_assignee_normalized")
    assert assignee is not None and assignee.operator == "contains" and assignee.value == "COCA COLA"

    state = _filter(parsed, "legal_state")
    assert state is not None and state.operator == "equals" and state.value == "Alive"


def case_8_three_filters(qu):
    """Three combined filters: granted + assignee + country."""
    parsed = qu.parse("granted Coca Cola bottle design patents in the US")
    assert "bottle design" in parsed.semantic_query

    status = _filter(parsed, "legal_status")
    assert status is not None and status.operator == "equals" and status.value == "Granted"

    assignee = _filter(parsed, "current_assignee_normalized")
    assert assignee is not None and assignee.operator == "contains" and assignee.value == "COCA COLA"

    country = _filter(parsed, "application_country")
    assert country is not None and country.operator == "equals" and country.value == "US"


def case_9_semantic_question_no_filters(qu):
    """Regression guard: a natural question must never spuriously pick up a filter."""
    query = "What patents describe a bottle with a removable cap and pressure control?"
    parsed = qu.parse(query)
    assert parsed.metadata_filters == []


def case_10_unsupported_filter_not_invented(qu):
    """Unsupported/ambiguous candidate filter is dropped, NEVER invents a field."""
    res = resolve_candidate_filter(meaning="company_headquarters", value="Atlanta")
    assert res is None or res.field in FIELD_MAPPING

    res_color = resolve_candidate_filter(meaning="bottle_color", value="red")
    assert res_color is None

    parsed = qu.parse("bottle design with red color owned by Coca Cola")
    for f in parsed.metadata_filters:
        assert f.field in FIELD_MAPPING
        assert f.field != "bottle_color"
        assert f.field != "company_headquarters"


def case_11_all_filters_exist_in_field_mapping(qu):
    """Every resolved filter must be in FIELD_MAPPING for a variety of queries."""
    queries = [
        "bottle design",
        "bottle design assigned to Coca Cola",
        "bottle design patents from the US",
        "active Coca Cola bottle patents from Japan",
        "bottle design patents filed after 2020",
    ]
    for q in queries:
        parsed = qu.parse(q)
        for f in parsed.metadata_filters:
            assert f.field in FIELD_MAPPING, f"Field '{f.field}' is not in FIELD_MAPPING allowlist!"


def case_12_natural_language_variations(qu):
    """Natural language variations for assignee, country, filing country."""
    p1 = qu.parse("bottle design patents owned by Coca Cola")
    f1 = _filter(p1, "current_assignee_normalized")
    assert f1 is not None and f1.value == "COCA COLA"

    p2 = qu.parse("bottle design patents from the US")
    f2 = _filter(p2, "application_country")
    assert f2 is not None and f2.value == "US"

    p3 = qu.parse("bottle design patents filed in Japan")
    f3 = _filter(p3, "application_country")
    assert f3 is not None and f3.value == "JP"


def case_13_llm_invalid_json_fallback(qu):
    """If LLM returns garbage, the rule-based fallback should still parse correctly."""
    # Test the rule-based parser directly (simulates LLM failure fallback)
    parsed = qu.parse_rule_based("bottle design patents assigned to Coca Cola in the US")
    assert "bottle design" in parsed.semantic_query

    assignee = _filter(parsed, "current_assignee_normalized")
    assert assignee is not None and assignee.value == "COCA COLA"

    country = _filter(parsed, "application_country")
    assert country is not None and country.value == "US"

    for f in parsed.metadata_filters:
        assert f.field in FIELD_MAPPING


def case_14_llm_unavailable_fallback(qu):
    """If LLM is unavailable, QueryUnderstanding(use_llm=False) uses rule-based parser."""
    qu_no_llm = QueryUnderstanding(use_llm=False)
    parsed = qu_no_llm.parse("bottle designs patented by Coca Cola in the US")

    assert "bottle design" in parsed.semantic_query
    assert len(parsed.metadata_filters) >= 1

    for f in parsed.metadata_filters:
        assert f.field in FIELD_MAPPING


def case_15_candidate_meaning_resolves_through_field_mapping(qu):
    """Candidate meaning -> resolve_candidate_filter -> FIELD_MAPPING field."""
    # "patented by" should resolve to current_assignee_normalized
    res1 = resolve_candidate_filter("patented by", "Coca Cola")
    assert res1 is not None
    assert res1.field == "current_assignee_normalized"
    assert res1.field in FIELD_MAPPING
    assert res1.value == "COCA COLA"

    # "in the US" should resolve to application_country
    res2 = resolve_candidate_filter("in the US", "US")
    assert res2 is not None
    assert res2.field == "application_country"
    assert res2.field in FIELD_MAPPING
    assert res2.value == "US"

    # "published in Japan" -> publication_country_code
    res3 = resolve_candidate_filter("published in Japan", "Japan")
    assert res3 is not None
    assert res3.field == "publication_country_code"
    assert res3.field in FIELD_MAPPING
    assert res3.value == "JP"

    # "published after" with year -> publication_year
    res4 = resolve_candidate_filter("published after", "2018")
    assert res4 is not None
    assert res4.field == "publication_year"
    assert res4.field in FIELD_MAPPING
    assert res4.operator == "gt"
    assert res4.value == 2018

    # unsupported meaning -> None
    res5 = resolve_candidate_filter("bottle_material", "glass")
    assert res5 is None

    res6 = resolve_candidate_filter("invented_database_field", "test")
    assert res6 is None


CASES = [
    ("Test 1  - pure semantic search", case_1_pure_semantic),
    ("Test 2  - assignee filter", case_2_assignee_filter),
    ("Test 3  - country filter (trailing idiom)", case_3_country_trailing_idiom),
    ("Test 4  - country filter (leading idiom)", case_4_country_leading_idiom),
    ("Test 5  - application year", case_5_application_year),
    ("Test 6  - publication country + year (gt)", case_6_country_plus_year_gt),
    ("Test 7  - active Coca Cola bottle patents", case_7_bare_assignee_plus_enum),
    ("Test 8  - three filters combined", case_8_three_filters),
    ("Test 9  - semantic question, no filters", case_9_semantic_question_no_filters),
    ("Test 10 - unsupported filter not invented", case_10_unsupported_filter_not_invented),
    ("Test 11 - all final filters in FIELD_MAPPING", case_11_all_filters_exist_in_field_mapping),
    ("Test 12 - natural language variations", case_12_natural_language_variations),
    ("Test 13 - LLM invalid JSON fallback", case_13_llm_invalid_json_fallback),
    ("Test 14 - LLM unavailable fallback", case_14_llm_unavailable_fallback),
    ("Test 15 - candidate meaning resolves through FIELD_MAPPING", case_15_candidate_meaning_resolves_through_field_mapping),
]


def main():
    qu = QueryUnderstanding()
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

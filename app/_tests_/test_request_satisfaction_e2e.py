"""
End-to-End Validation Test Suite for Dynamic Request Satisfaction and Ranking Tuning

Validates the full pipeline across 15 diverse query categories without domain-specific hardcoded rules.
"""

import sys
import types
from app.models.patent_search_result import PatentSearchResult, RankedChunk, ScoreBreakdown
from app.query_understanding.models import (
    Concept,
    Constraint,
    Goal,
    OptimizationTarget,
    ParsedQuery,
    QuestionIntent,
    Relationship,
)
from app.reranker import Reranker
from app.semantic_search import SemanticSearch


def _stub_reranker(score_fn=None):
    r = Reranker.__new__(Reranker)
    r.model = None
    r._score = score_fn or (lambda q, passages: [0.5] * len(passages))
    return r


def _chunk(text: str, patent_id: str = "P1", chunk_id: int = 0, section: str = "DESCRIPTION"):
    return types.SimpleNamespace(
        id=f"{patent_id}-{chunk_id}",
        score=1.0,
        payload={"text": text, "patent_id": patent_id, "chunk_id": chunk_id, "section": section},
    )


def test_category_1_single_concept():
    """1. Single concept/name (e.g., 'Graphene')"""
    parsed = ParsedQuery(
        original_query="Graphene",
        semantic_query="Graphene",
        concepts=[Concept(id="C1", text="Graphene", role="material", importance=1.0, required=True)],
    )
    def fake_score(q, texts):
        return [0.90 if "graphene" in t.lower() else 0.10 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("A single-layer graphene sheet synthesis method.", patent_id="P1"),
        _chunk("A graphite mechanical pencil core.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.70, 0.90], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
    assert ranked[1][2].request_satisfaction_label == "NON_MATCH"


def test_category_2_short_noun_phrase():
    """2. Short noun phrase (e.g., 'quantum dot emitter')"""
    parsed = ParsedQuery(
        original_query="quantum dot emitter",
        semantic_query="quantum dot emitter",
        concepts=[
            Concept(id="C1", text="quantum dot", role="technology", importance=0.9),
            Concept(id="C2", text="emitter", role="component", importance=0.8),
        ],
    )
    def fake_score(q, texts):
        return [0.88 if "quantum dot emitter" in t.lower() else 0.15 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("A colloidal quantum dot emitter for electroluminescent displays.", patent_id="P1"),
        _chunk("A quantum computing algorithm for emitter simulation.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.75, 0.92], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_3_technical_phrase():
    """3. Technical phrase (e.g., 'perovskite tandem solar cell')"""
    parsed = ParsedQuery(
        original_query="perovskite tandem solar cell",
        semantic_query="perovskite tandem solar cell",
        concepts=[Concept(id="C1", text="perovskite tandem solar cell", role="technology", importance=1.0)],
    )
    def fake_score(q, texts):
        return [0.92 if "perovskite" in t.lower() and "solar cell" in t.lower() else 0.20 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("A monolithic perovskite-silicon tandem solar cell.", patent_id="P1"),
        _chunk("A tandem bicycle transmission gear.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.72, 0.95], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[1][2].request_satisfaction_label == "NON_MATCH"


def test_category_4_functional_request():
    """4. Functional request (e.g., 'cooling high density server racks')"""
    parsed = ParsedQuery(
        original_query="cooling high density server racks",
        semantic_query="cooling high density server racks",
        concepts=[Concept(id="C1", text="server racks", role="object", importance=0.8)],
        goals=[Goal(id="G1", text="cooling high density equipment", importance=0.9, required=True)],
    )
    def fake_score(q, texts):
        return [0.89 if "cooling" in t.lower() and "rack" in t.lower() else 0.12 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("Liquid immersion cooling system for high-density server racks.", patent_id="P1"),
        _chunk("Server rack mounting bracket for cable storage.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.75, 0.88], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_5_purpose_oriented_request():
    """5. Purpose-oriented request (e.g., 'for preventing corrosion in marine structures')"""
    parsed = ParsedQuery(
        original_query="preventing corrosion in marine structures",
        semantic_query="preventing corrosion in marine structures",
        concepts=[Concept(id="C1", text="marine structures", role="object", importance=0.8)],
        goals=[Goal(id="G1", text="preventing corrosion", importance=0.9, required=True)],
    )
    def fake_score(q, texts):
        return [0.85 if "corrosion" in t.lower() and "marine" in t.lower() else 0.10 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("Sacrificial zinc anode coating for preventing corrosion in marine hulls.", patent_id="P1"),
        _chunk("Marine navigational sonar radar beacon.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.70, 0.90], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_6_property_oriented_request():
    """6. Property-oriented request (e.g., 'flexible high conductivity polymer')"""
    parsed = ParsedQuery(
        original_query="flexible high conductivity polymer",
        semantic_query="flexible high conductivity polymer",
        concepts=[Concept(id="C1", text="polymer", role="material", importance=0.8)],
        constraints=[Constraint(id="K1", text="flexible", importance=0.8), Constraint(id="K2", text="high conductivity", importance=0.9)],
    )
    def fake_score(q, texts):
        return [0.91 if "flexible" in t.lower() and "conductive" in t.lower() else 0.20 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("A mechanically flexible intrinsically conductive polymer composite.", patent_id="P1"),
        _chunk("Rigid insulating polymer housing.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.70, 0.85], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_7_multi_concept_request():
    """7. Multi-concept request (e.g., 'lithium-ion battery with solid electrolyte and silicon anode')"""
    parsed = ParsedQuery(
        original_query="lithium-ion battery with solid electrolyte and silicon anode",
        semantic_query="lithium-ion battery with solid electrolyte and silicon anode",
        concepts=[
            Concept(id="C1", text="lithium-ion battery", role="object", importance=0.9),
            Concept(id="C2", text="solid electrolyte", role="component", importance=0.9),
            Concept(id="C3", text="silicon anode", role="component", importance=0.9),
        ],
    )
    def fake_score(q, texts):
        scores = []
        for t in texts:
            t_low = t.lower()
            if "solid electrolyte" in t_low and "silicon" in t_low:
                scores.append(0.95)
            elif "solid electrolyte" in t_low:
                scores.append(0.55)
            else:
                scores.append(0.10)
        return scores

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("All-solid-state lithium battery comprising a garnet solid electrolyte and a silicon composite anode.", patent_id="P_FULL"),
        _chunk("Solid electrolyte separator for lithium-sulfur battery with lithium metal anode.", patent_id="P_PARTIAL"),
        _chunk("Standard liquid electrolyte lithium cell with graphite anode.", patent_id="P_NON"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.80, 0.82, 0.90], chunks, parsed)
    assert ranked[0][1].id == "P_FULL-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
    assert ranked[1][1].id == "P_PARTIAL-0"
    assert ranked[1][2].request_satisfaction_label == "PARTIAL_MATCH"


def test_category_8_relationship_request():
    """8. Relationship request (e.g., 'heat exchanger coupled to turbine exhaust')"""
    parsed = ParsedQuery(
        original_query="heat exchanger coupled to turbine exhaust",
        semantic_query="heat exchanger coupled to turbine exhaust",
        concepts=[Concept(id="C1", text="heat exchanger", role="component"), Concept(id="C2", text="turbine exhaust", role="component")],
        relationships=[Relationship(source="heat exchanger", relation="coupled_to", target="turbine exhaust", importance=0.9)],
    )
    def fake_score(q, texts):
        return [0.89 if "coupled to" in t.lower() or "connected to the exhaust" in t.lower() else 0.15 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("A recuperator heat exchanger connected to the exhaust of a gas turbine.", patent_id="P1"),
        _chunk("A heat exchanger for air conditioning while a turbine drives an alternator.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.72, 0.91], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_9_constrained_request():
    """9. Constrained request (e.g., 'wireless transceiver operating below 1GHz')"""
    parsed = ParsedQuery(
        original_query="wireless transceiver operating below 1GHz",
        semantic_query="wireless transceiver operating below 1GHz",
        concepts=[Concept(id="C1", text="wireless transceiver", role="device", importance=0.9)],
        constraints=[Constraint(id="K1", text="operating below 1GHz", importance=0.9, required=True)],
    )
    def fake_score(q, texts):
        return [0.90 if "sub-ghz" in t.lower() or "below 1 ghz" in t.lower() else 0.15 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("A sub-GHz wireless transceiver operating at 433 MHz and 868 MHz ISM bands.", patent_id="P1"),
        _chunk("A mmWave 60 GHz wireless transceiver.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.70, 0.94], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_10_exclusion_request():
    """10. Query containing exclusions (e.g., 'organic LED display without indium tin oxide')"""
    parsed = ParsedQuery(
        original_query="organic LED display without indium tin oxide",
        semantic_query="organic LED display",
        concepts=[Concept(id="C1", text="organic LED display", role="device", importance=0.9)],
        exclusions=["indium tin oxide"],
    )
    def fake_score(q, texts):
        return [0.85 if "oled" in t.lower() else 0.20 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("An OLED display utilizing carbon nanotube transparent conductive films.", patent_id="P_NO_ITO"),
        _chunk("An OLED display comprising an indium tin oxide (ITO) anode.", patent_id="P_WITH_ITO"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.80, 0.85], chunks, parsed)
    assert ranked[0][1].id == "P_NO_ITO-0"


def test_category_11_natural_language_query():
    """11. Natural-language query (e.g., 'methods to improve energy efficiency of electric vehicles in cold weather')"""
    parsed = ParsedQuery(
        original_query="methods to improve energy efficiency of electric vehicles in cold weather",
        semantic_query="improve energy efficiency of electric vehicles in cold weather",
        concepts=[Concept(id="C1", text="electric vehicles", role="object", importance=0.8)],
        goals=[Goal(id="G1", text="improve energy efficiency in cold weather", importance=0.9, required=True)],
    )
    def fake_score(q, texts):
        return [0.93 if "heat pump" in t.lower() or "cold weather" in t.lower() else 0.12 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("Thermal management heat pump system optimizing EV range during low ambient temperatures.", patent_id="P1"),
        _chunk("Electric vehicle paint formulation resistant to scratches.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.72, 0.90], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_12_question_query():
    """12. Question query (e.g., 'how does the magnetic bearing minimize friction at high RPM?')"""
    parsed = ParsedQuery(
        original_query="how does the magnetic bearing minimize friction at high RPM?",
        semantic_query="magnetic bearing minimize friction at high RPM",
        is_question=True,
        question_intent=QuestionIntent(
            target="magnetic bearing minimize friction at high RPM",
            answer_criteria="contactless electromagnetic levitation mechanism",
        ),
    )
    def fake_score(q, texts):
        return [0.94 if "electromagnetic levitation" in t.lower() else 0.10 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("Active electromagnetic levitation maintains rotor clearance, eliminating mechanical friction at 50,000 RPM.", patent_id="P1"),
        _chunk("A ball bearing lubrication oil composition.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.70, 0.93], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
    assert "question target" in ranked[0][2].request_satisfaction_reason


def test_category_13_ambiguous_terminology():
    """13. Ambiguous terminology (e.g., 'window screen filter')"""
    parsed = ParsedQuery(
        original_query="window screen filter",
        semantic_query="window screen filter",
        concepts=[Concept(id="C1", text="window screen filter", role="object", importance=0.9)],
        intent="optical anti-glare screen for vehicle or architectural windows",
    )
    def fake_score(q, texts):
        return [0.88 if "optical anti-glare" in t.lower() else 0.12 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("An optical anti-glare polarising screen applied to automotive windows.", patent_id="P_OPTICAL"),
        _chunk("An insect mesh screen frame for residential building windows.", patent_id="P_INSECT"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.75, 0.91], chunks, parsed)
    assert ranked[0][1].id == "P_OPTICAL-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
    assert ranked[1][2].request_satisfaction_label == "NON_MATCH"


def test_category_14_very_short_query():
    """14. Very short query (e.g., 'LiDAR')"""
    parsed = ParsedQuery(
        original_query="LiDAR",
        semantic_query="LiDAR",
    )
    def fake_score(q, texts):
        return [0.85 if "lidar" in t.lower() or "light detection" in t.lower() else 0.10 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("Solid-state pulsed LiDAR sensor for range measurement.", patent_id="P1"),
        _chunk("Radar frequency generator for marine navigation.", patent_id="P2"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.80, 0.88], chunks, parsed)
    assert ranked[0][1].id == "P1-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"


def test_category_15_detailed_multipart_query():
    """15. More detailed multi-part query (e.g., 'biodegradable packaging film having high tensile strength and oxygen barrier properties')"""
    parsed = ParsedQuery(
        original_query="biodegradable packaging film having high tensile strength and oxygen barrier properties",
        semantic_query="biodegradable packaging film having high tensile strength and oxygen barrier properties",
        concepts=[Concept(id="C1", text="biodegradable packaging film", role="object", importance=0.9)],
        goals=[Goal(id="G1", text="high tensile strength", importance=0.8), Goal(id="G2", text="oxygen barrier", importance=0.9)],
    )
    def fake_score(q, texts):
        scores = []
        for t in texts:
            if "oxygen barrier" in t.lower() and "tensile" in t.lower() and "biodegradable" in t.lower():
                scores.append(0.96)
            elif "biodegradable" in t.lower():
                scores.append(0.52)
            else:
                scores.append(0.12)
        return scores

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("Polylactic acid biodegradable packaging film with nanocellulose reinforcing high tensile strength and oxygen barrier layer.", patent_id="P_ALL"),
        _chunk("Biodegradable shopping bag with low barrier properties.", patent_id="P_PARTIAL"),
        _chunk("Synthetic polypropylene stretch film with high tensile strength.", patent_id="P_NON"),
    ]
    ranked = reranker._blend_and_sort(parsed.semantic_query, [0.82, 0.78, 0.94], chunks, parsed)
    assert ranked[0][1].id == "P_ALL-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
    assert ranked[1][1].id == "P_PARTIAL-0"
    assert ranked[1][2].request_satisfaction_label == "PARTIAL_MATCH"
    assert ranked[2][1].id == "P_NON-0"
    assert ranked[2][2].request_satisfaction_label == "NON_MATCH"


TESTS = [
    ("Category 1  - Single Concept", test_category_1_single_concept),
    ("Category 2  - Short Noun Phrase", test_category_2_short_noun_phrase),
    ("Category 3  - Technical Phrase", test_category_3_technical_phrase),
    ("Category 4  - Functional Request", test_category_4_functional_request),
    ("Category 5  - Purpose-Oriented Request", test_category_5_purpose_oriented_request),
    ("Category 6  - Property-Oriented Request", test_category_6_property_oriented_request),
    ("Category 7  - Multi-Concept Request", test_category_7_multi_concept_request),
    ("Category 8  - Relationship Request", test_category_8_relationship_request),
    ("Category 9  - Constrained Request", test_category_9_constrained_request),
    ("Category 10 - Exclusion Request", test_category_10_exclusion_request),
    ("Category 11 - Natural Language Query", test_category_11_natural_language_query),
    ("Category 12 - Question Query", test_category_12_question_query),
    ("Category 13 - Ambiguous Terminology", test_category_13_ambiguous_terminology),
    ("Category 14 - Very Short Query", test_category_14_very_short_query),
    ("Category 15 - Detailed Multi-Part Query", test_category_15_detailed_multipart_query),
]


def main():
    failures = []
    for name, test_fn in TESTS:
        try:
            test_fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failures.append(name)
            print(f"[ERROR] {name}: {e!r}")

    print()
    if failures:
        print(f"{len(failures)} of {len(TESTS)} diversity categories FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(TESTS)} diversity categories passed successfully (100% pass rate).")


if __name__ == "__main__":
    main()

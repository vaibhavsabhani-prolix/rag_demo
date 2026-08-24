"""
Test cases for the dynamic Reranker (app/reranker.py).

Tier 1 (below, no model/network calls) - pure deterministic logic:
weight computation, has_requirements_structure, gating/blend math -
via a Reranker whose _score() is stubbed out.

Tier 2 - real local CrossEncoder scoring for the five acceptance-test
query classes from the redesign plan, guarded so a missing/unavailable
model skips gracefully rather than failing the whole suite.

Run with:
    ./.venv/bin/python -m app._tests_.test_reranker
"""

import sys
import types

from app.models.patent_search_result import ScoreBreakdown
from app.query_understanding.models import (
    Concept,
    Constraint,
    Goal,
    OptimizationTarget,
    ParsedQuery,
    RankingWeights,
    Relationship,
)
from app.reranker import Reranker, _compute_weights

# ==================================================================
# Builders
# ==================================================================


def _chunk(text: str, patent_id: str = "P1", chunk_id: int = 0):
    return types.SimpleNamespace(
        id=f"{patent_id}-{chunk_id}",
        score=1.0,
        payload={"text": text, "patent_id": patent_id, "chunk_id": chunk_id},
    )


def _parsed(**kwargs) -> ParsedQuery:
    kwargs.setdefault("original_query", "q")
    kwargs.setdefault("semantic_query", "q")
    return ParsedQuery(**kwargs)


def _stub_reranker(score_fn):
    """
    A Reranker instance with __init__ skipped (no model load, no
    network) - _score is replaced with *score_fn(query, texts)*.
    """

    r = object.__new__(Reranker)
    r.use_remote = False
    r._score = score_fn
    return r


# ==================================================================
# Tier 1 - weight computation
# ==================================================================


def case_1_simple_query_weights_favor_semantic(_):
    """A query with just a couple of concepts (no rel/opt/exclusions) keeps semantic dominant."""
    parsed = _parsed(
        concepts=[Concept(id="C1", text="bottle design", role="object", importance=0.8)]
    )
    w = _compute_weights(parsed)
    assert w["semantic"] + w["structured"] >= 0.55
    assert w["relationship"] == 0.0
    assert w["optimization"] == 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-9


def case_2_optimization_only_query_does_not_break_floor(_):
    """A pure-optimization query (no Concept/Goal/Constraint at all) still keeps the semantic floor."""
    parsed = _parsed(
        optimization=[
            OptimizationTarget(
                id="O1",
                property="power consumption",
                direction="minimize",
                importance=0.9,
            )
        ]
    )
    assert parsed.has_requirements_structure is True  # the fix under test
    w = _compute_weights(parsed)
    assert w["semantic"] + w["structured"] >= 0.55
    assert w["optimization"] == max(w["optimization"], 0.0)
    assert w["optimization"] <= 0.20 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def case_3_relationship_weight_is_clamped(_):
    """A high LLM-provided relationship_satisfaction is clamped to MAX_SECONDARY_WEIGHT, not trusted raw."""
    parsed = _parsed(
        concepts=[Concept(id="C1", text="camera", role="technology", importance=0.7)],
        relationships=[
            Relationship(
                source="camera",
                relation="used_for",
                target="defect detection",
                importance=1.0,
            )
        ],
        ranking_weights=RankingWeights(
            semantic_relevance=1.0, relationship_satisfaction=0.9
        ),
    )
    w = _compute_weights(parsed)
    assert w["relationship"] <= 0.20 + 1e-9
    assert w["semantic"] + w["structured"] >= 0.55


def case_4_llm_zero_semantic_weight_is_ignored(_):
    """Even if the LLM outputs semantic_relevance=0.0, application code keeps semantic dominant."""
    parsed = _parsed(
        concepts=[Concept(id="C1", text="widget", role="object", importance=0.9)],
        relationships=[
            Relationship(source="a", relation="controls", target="b", importance=1.0)
        ],
        ranking_weights=RankingWeights(
            semantic_relevance=0.0, relationship_satisfaction=1.0
        ),
    )
    w = _compute_weights(parsed)
    assert w["semantic"] + w["structured"] >= 0.55, (
        "LLM's semantic_relevance=0.0 must not be trusted raw"
    )


def case_5_weights_always_sum_to_one(_):
    scenarios = [
        _parsed(),
        _parsed(concepts=[Concept(id="C1", text="x", role="object", importance=1.0)]),
        _parsed(
            concepts=[Concept(id="C1", text="x", role="object", importance=1.0)],
            goals=[Goal(id="G1", text="y", importance=1.0, required=True)],
            constraints=[Constraint(id="K1", text="z", importance=1.0, required=True)],
            optimization=[
                OptimizationTarget(
                    id="O1", property="p", direction="maximize", importance=1.0
                )
            ],
            relationships=[
                Relationship(source="a", relation="r", target="b", importance=1.0)
            ],
            exclusions=["nope"],
        ),
    ]
    for parsed in scenarios:
        w = _compute_weights(parsed)
        assert abs(sum(w.values()) - 1.0) < 1e-9, w


# ==================================================================
# Tier 1 - has_requirements_structure
# ==================================================================


def case_6_has_requirements_structure_empty_is_false(_):
    assert _parsed().has_requirements_structure is False


def case_7_has_requirements_structure_optimization_only_is_true(_):
    parsed = _parsed(
        optimization=[
            OptimizationTarget(id="O1", property="cost", direction="minimize")
        ]
    )
    assert parsed.has_requirements_structure is True


# ==================================================================
# Tier 1 - stubbed-model gating/blend logic
# ==================================================================


def case_8_optimization_gating_suppresses_offtopic_noise(_):
    """
    A chunk that doesn't discuss the property at all should land near
    neutral (0.5), not get pulled toward either direction by noise.
    """
    reranker = _stub_reranker(
        lambda query, texts: [0.05 for _ in texts]
    )  # both aligned & opposed score low
    target = OptimizationTarget(
        id="O1", property="power consumption", direction="minimize", importance=1.0
    )
    scores = reranker._score_optimization(
        [target], fine_indices=[0], fine_texts=["irrelevant text"]
    )
    assert abs(scores[0] - 0.5) < 0.05, scores


def case_9_optimization_gating_rewards_aligned_direction(_):
    """A chunk clearly discussing the aligned direction should score above neutral."""

    def fake_score(query, texts):
        # "reduces"/"low" phrasing scores high, "increases"/"high" phrasing scores low
        if "reduces" in query or "low" in query:
            return [0.9 for _ in texts]
        return [0.1 for _ in texts]

    reranker = _stub_reranker(fake_score)
    target = OptimizationTarget(
        id="O1", property="power consumption", direction="minimize", importance=1.0
    )
    scores = reranker._score_optimization(
        [target], fine_indices=[0], fine_texts=["reduces power consumption"]
    )
    assert scores[0] > 0.5, scores


def case_10_optimization_gating_penalizes_opposed_direction(_):
    def fake_score(query, texts):
        if "increases" in query or "high" in query:
            return [0.9 for _ in texts]
        return [0.1 for _ in texts]

    reranker = _stub_reranker(fake_score)
    target = OptimizationTarget(
        id="O1", property="power consumption", direction="minimize", importance=1.0
    )
    scores = reranker._score_optimization(
        [target], fine_indices=[0], fine_texts=["increases power consumption"]
    )
    assert scores[0] < 0.5, scores


def case_11_exclusion_literal_hit_outweighs_semantic_only(_):
    """A literal verbatim exclusion match must be treated as at least as strong as any semantic-only signal."""
    reranker = _stub_reranker(
        lambda query, texts: [0.2 for _ in texts]
    )  # low semantic signal
    results = [_chunk("this uses lithium directly")]
    penalties = reranker._score_exclusions(
        ["lithium"],
        fine_indices=[0],
        fine_texts=["this uses lithium directly"],
        results=results,
    )
    assert penalties[0] == 1.0, penalties


def case_12_exclusion_semantic_paraphrase_still_detected(_):
    """A paraphrased exclusion mention (no literal term) is still caught via the semantic probe."""
    reranker = _stub_reranker(
        lambda query, texts: [0.85 for _ in texts]
    )  # high semantic signal, no literal term
    results = [_chunk("the cell chemistry relies on an alkali-metal-ion compound")]
    penalties = reranker._score_exclusions(
        ["lithium"], fine_indices=[0], fine_texts=["..."], results=results
    )
    assert penalties[0] == 0.85, penalties


def case_13_relationship_scoring_is_importance_weighted(_):
    # _build_relationship_sentence renders "{source} {relation} {target}" -
    # match on the relation text embedded in that templated sentence.
    def fake_score(query, texts):
        return [1.0 if "important relation" in query else 0.0 for _ in texts]

    reranker = _stub_reranker(fake_score)
    relationships = [
        Relationship(
            source="a", relation="important relation", target="b", importance=1.0
        ),
        Relationship(source="c", relation="minor relation", target="d", importance=0.1),
    ]
    scores = reranker._score_relationships(
        relationships, fine_indices=[0], fine_texts=["text"]
    )
    assert scores[0] > 0.85, scores  # dominated by the high-importance relationship


def case_14_pure_fallback_matches_semantic_order(_):
    """No requirements structure -> rerank() falls back to a pure semantic-score sort."""
    reranker = _stub_reranker(lambda query, texts: [0.2, 0.9, 0.5])
    results = [
        _chunk("a", chunk_id=0),
        _chunk("b", chunk_id=1),
        _chunk("c", chunk_id=2),
    ]
    ranked = reranker.rerank("query", results, requirements=None)
    assert [r[1].id for r in ranked] == [results[1].id, results[2].id, results[0].id]
    assert all(
        r[2].final_score == r[0] for r in ranked
    )  # ScoreBreakdown always populated


def case_15_non_fine_stage_chunks_keep_pure_semantic_score(_):
    """Chunks outside the RERANK_FINE_STAGE_TOP_N cutoff must not get an arbitrary scaling/penalty."""
    import app.reranker as reranker_module

    original_top_n = reranker_module.RERANK_FINE_STAGE_TOP_N
    reranker_module.RERANK_FINE_STAGE_TOP_N = 1
    try:
        semantic_scores = [0.9, 0.5, 0.1]
        reranker = _stub_reranker(lambda query, texts: semantic_scores[: len(texts)])
        results = [
            _chunk("on-target", chunk_id=0),
            _chunk("mid", chunk_id=1),
            _chunk("low", chunk_id=2),
        ]
        parsed = _parsed(
            concepts=[Concept(id="C1", text="on-target", role="object", importance=0.9)]
        )

        ranked = reranker._blend_and_sort("q", semantic_scores, results, parsed)
        by_id = {chunk.id: score for score, chunk, _ in ranked}

        # "mid" and "low" fell outside the (TOP_N=1) fine stage - their
        # final_score must equal their raw coarse semantic score exactly.
        assert by_id[results[1].id] == 0.5
        assert by_id[results[2].id] == 0.1
    finally:
        reranker_module.RERANK_FINE_STAGE_TOP_N = original_top_n


# ==================================================================
# Tier 2 - real local CrossEncoder, the five acceptance-test query
# classes from the redesign plan. Guarded: if the local model can't
# be loaded in this environment, these cases are skipped rather than
# failing the whole suite (matching the project's lenient-fallback style).
# ==================================================================

_REAL_RERANKER = None
_REAL_RERANKER_ERROR = None


def _get_real_reranker():
    global _REAL_RERANKER, _REAL_RERANKER_ERROR
    if _REAL_RERANKER is not None or _REAL_RERANKER_ERROR is not None:
        return _REAL_RERANKER
    try:
        _REAL_RERANKER = Reranker(use_remote=False)
    except Exception as e:
        _REAL_RERANKER_ERROR = e
    return _REAL_RERANKER


def _assert_on_target_wins(
    case_name,
    parsed,
    on_target_text,
    off_target_text,
    on_target_semantic,
    off_target_semantic,
):
    reranker = _get_real_reranker()
    if reranker is None:
        print(
            f"[SKIP] {case_name} - local reranker model unavailable: {_REAL_RERANKER_ERROR}"
        )
        return

    on_chunk = _chunk(on_target_text, patent_id="ON")
    off_chunk = _chunk(off_target_text, patent_id="OFF")
    results = [on_chunk, off_chunk]

    # Force the coarse semantic scores to mimic a plausible cross-encoder
    # mistake (off-target scores marginally higher) - the fine-stage
    # blend must still recover and favor the genuinely on-target chunk.
    real_score = reranker._score
    reranker._score = lambda query, texts, _real=real_score, _q=parsed.semantic_query: (
        [on_target_semantic, off_target_semantic]
        if query == _q
        else _real(query, texts)
    )
    try:
        ranked = reranker._blend_and_sort(
            parsed.semantic_query,
            [on_target_semantic, off_target_semantic],
            results,
            parsed,
        )
    finally:
        reranker._score = real_score

    order = [chunk.id for _, chunk, _ in ranked]
    assert order[0] == "ON-0", (
        f"{case_name}: expected on-target chunk to win, got order {order}"
    )
    print(f"[OK]   {case_name} -> on-target chunk correctly outranked off-target chunk")


def case_16_defect_detection_relationship_query(_):
    parsed = _parsed(
        semantic_query="automatically detecting defects in manufactured products using cameras and AI",
        concepts=[
            Concept(
                id="C1", text="manufactured products", role="object", importance=0.7
            ),
            Concept(
                id="C2",
                text="camera",
                role="technology",
                importance=0.8,
                semantic_variants=["imaging sensor"],
            ),
            Concept(
                id="C3",
                text="AI",
                role="technology",
                importance=0.8,
                semantic_variants=["neural network"],
            ),
        ],
        goals=[
            Goal(
                id="G1",
                text="automatically detect defects",
                importance=0.9,
                required=True,
                keywords=["identifies defective articles", "detects defects"],
            )
        ],
        relationships=[
            Relationship(
                source="camera",
                relation="used_for",
                target="defect detection",
                importance=0.9,
            )
        ],
    )
    _assert_on_target_wins(
        "defect detection (relationship)",
        parsed,
        on_target_text="An imaging sensor captures production-line images and a neural network identifies defective articles.",
        off_target_text="A camera monitors ambient temperature while defects are logged by a separate, unrelated inspection process.",
        on_target_semantic=0.55,
        off_target_semantic=0.60,
    )


def case_17_power_consumption_minimize_query(_):
    parsed = _parsed(
        semantic_query="reducing power consumption in semiconductor devices",
        concepts=[
            Concept(
                id="C1", text="semiconductor devices", role="object", importance=0.8
            )
        ],
        optimization=[
            OptimizationTarget(
                id="O1",
                property="power consumption",
                direction="minimize",
                importance=0.9,
            )
        ],
    )
    _assert_on_target_wins(
        "power consumption (optimization: minimize)",
        parsed,
        on_target_text="A semiconductor device with a gating circuit that substantially reduces power consumption during idle states.",
        off_target_text="A semiconductor device optimized for maximum throughput that results in significantly increased power consumption.",
        on_target_semantic=0.55,
        off_target_semantic=0.60,
    )


def case_18_plant_meat_taste_query(_):
    parsed = _parsed(
        semantic_query="improve the taste of plant-based meat",
        concepts=[
            Concept(
                id="C1",
                text="plant-based meat",
                role="object",
                importance=0.9,
                semantic_variants=["meat analogue", "meat substitute"],
            )
        ],
        goals=[
            Goal(
                id="G1",
                text="improve taste",
                importance=0.9,
                required=True,
                keywords=["improved palatability", "enhanced flavor profile"],
            )
        ],
    )
    _assert_on_target_wins(
        "plant meat taste (goal, no literal overlap)",
        parsed,
        on_target_text="A meat analogue composition incorporating a flavor precursor that yields an improved palatability and enhanced flavor profile upon cooking.",
        off_target_text="A packaging film for storing food products with improved moisture barrier properties.",
        on_target_semantic=0.50,
        off_target_semantic=0.55,
    )


def case_19_sweetness_sugar_constraint_query(_):
    parsed = _parsed(
        semantic_query="natural ways to improve sweetness while keeping sugar low",
        goals=[
            Goal(
                id="G1",
                text="increase sweetness",
                importance=0.9,
                required=True,
                keywords=["enhanced sweetness", "sweetening effect"],
            )
        ],
        constraints=[
            Constraint(
                id="K1",
                text="keep sugar low",
                importance=0.9,
                required=True,
                keywords=["reduced sugar content", "low-sugar"],
            )
        ],
    )
    _assert_on_target_wins(
        "sweetness + low-sugar (goal + constraint)",
        parsed,
        on_target_text="A natural sweetening composition that provides an enhanced sweetening effect while achieving a reduced sugar content.",
        off_target_text="A natural sweetening composition that provides an enhanced sweetening effect through additional added sugar.",
        on_target_semantic=0.55,
        off_target_semantic=0.58,
    )


def case_20_vacuum_pressure_dual_optimization_query(_):
    parsed = _parsed(
        semantic_query="vacuum pressure machine with low electricity consumption and best performance",
        concepts=[
            Concept(
                id="C1", text="vacuum pressure machine", role="object", importance=0.9
            )
        ],
        optimization=[
            OptimizationTarget(
                id="O1",
                property="electricity consumption",
                direction="minimize",
                importance=0.8,
            ),
            OptimizationTarget(
                id="O2", property="performance", direction="maximize", importance=0.8
            ),
        ],
    )
    _assert_on_target_wins(
        "vacuum pressure machine (two simultaneous optimization targets)",
        parsed,
        on_target_text="A vacuum pressure machine featuring a motor design that reduces electricity consumption while achieving increased performance output.",
        off_target_text="A vacuum pressure machine featuring a motor design that increases electricity consumption while achieving decreased performance output.",
        on_target_semantic=0.55,
        off_target_semantic=0.58,
    )


def case_21_request_satisfaction_labels_and_breakdown(_):
    """Verify request satisfaction labels (DIRECT_MATCH, PARTIAL_MATCH, NON_MATCH) and breakdown observability."""

    def fake_score(query, texts):
        if "direct" in texts[0]:
            return [0.85]
        elif "partial" in texts[0]:
            return [0.50]
        else:
            return [0.10]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        concepts=[
            Concept(id="C1", text="television display", role="object", importance=1.0)
        ]
    )

    res_direct = reranker._score_request_satisfaction(
        parsed, "television display", [0], ["direct television display context"]
    )
    assert res_direct[0][1] == "DIRECT_MATCH"
    assert res_direct[0][0] == 0.85

    res_partial = reranker._score_request_satisfaction(
        parsed, "television display", [0], ["partial display context"]
    )
    assert res_partial[0][1] == "PARTIAL_MATCH"
    assert res_partial[0][0] == 0.50

    res_non = reranker._score_request_satisfaction(
        parsed, "television display", [0], ["insect screen context"]
    )
    assert res_non[0][1] == "NON_MATCH"
    assert res_non[0][0] == 0.10


def case_22_request_satisfaction_penalizes_non_match_over_lower_semantic_direct(_):
    """
    Candidate A: high semantic similarity (0.85), but NON_MATCH (0.10)
    Candidate B: moderate semantic similarity (0.55), but DIRECT_MATCH (0.90)
    Candidate B MUST outrank Candidate A.
    """

    def fake_score(query, texts):
        if "specifically directed to" in query:
            # satisfaction probe
            scores = []
            for t in texts:
                if "insect screen" in t:
                    scores.append(0.08)  # NON_MATCH
                else:
                    scores.append(0.88)  # DIRECT_MATCH
            return scores
        # Coarse semantic scores
        return [0.85, 0.55]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        semantic_query="television display",
        concepts=[
            Concept(
                id="C1",
                text="television display",
                role="object",
                importance=1.0,
                required=True,
            )
        ],
    )
    chunk_a = _chunk(
        "An insect screen mesh for window frames", patent_id="PA", chunk_id=0
    )
    chunk_b = _chunk(
        "An active-matrix television display panel", patent_id="PB", chunk_id=0
    )
    results = [chunk_a, chunk_b]

    ranked = reranker._blend_and_sort(
        parsed.semantic_query, [0.85, 0.55], results, parsed
    )
    assert ranked[0][1].id == "PB-0", (
        f"Expected DIRECT_MATCH chunk_b to win, got {ranked[0][1].id}"
    )
    assert ranked[1][1].id == "PA-0"

    b_breakdown = ranked[0][2]
    a_breakdown = ranked[1][2]
    assert b_breakdown.request_satisfaction_label == "DIRECT_MATCH"
    assert b_breakdown.request_satisfaction_score >= 0.65
    assert a_breakdown.request_satisfaction_label == "NON_MATCH"
    assert a_breakdown.request_satisfaction_score < 0.35
    assert "NON_MATCH" in a_breakdown.request_satisfaction_reason


def case_23_question_query_satisfaction_evaluates_target(_):
    """Verify question queries dynamically evaluate question intent target satisfaction."""
    from app.query_understanding.models import QuestionIntent

    def fake_score(query, texts):
        assert "calibrates temperature" in query
        return [0.90 if "calibration curve" in t else 0.15 for t in texts]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        original_query="How does the sensor calibrate temperature?",
        semantic_query="sensor calibrating temperature",
        is_question=True,
        question_intent=QuestionIntent(
            is_question=True,
            target="sensor calibrates temperature",
            expected_answer_type="method",
            answer_criteria="describes temperature calibration procedure",
        ),
    )
    chunk_answer = _chunk(
        "The sensor calculates a temperature offset using a polynomial calibration curve."
    )
    chunk_offtopic = _chunk(
        "A sensor is mounted near the heating element inside the vehicle engine."
    )

    satisfaction = reranker._score_request_satisfaction(
        parsed,
        parsed.semantic_query,
        [0, 1],
        [chunk_answer.payload["text"], chunk_offtopic.payload["text"]],
    )
    assert satisfaction[0][1] == "DIRECT_MATCH"
    assert satisfaction[1][1] == "NON_MATCH"


def case_24_multipart_request_satisfaction(_):
    """Multi-part requirements (concepts + goals + constraints) are evaluated dynamically."""

    def fake_score(query, texts):
        scores = []
        for t in texts:
            if "high yield" in t and "sub-zero" in t:
                scores.append(0.92)  # Fulfills all parts -> DIRECT_MATCH
            elif "high yield" in t:
                scores.append(0.52)  # Partial fulfillment -> PARTIAL_MATCH
            else:
                scores.append(0.12)  # Irrelevant -> NON_MATCH
        return scores

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        concepts=[
            Concept(id="C1", text="catalytic synthesis", role="process", importance=0.9)
        ],
        goals=[Goal(id="G1", text="high yield", importance=0.9, required=True)],
        constraints=[
            Constraint(
                id="K1", text="sub-zero temperature", importance=0.8, required=True
            )
        ],
    )
    chunks = [
        _chunk(
            "A catalytic synthesis method achieving high yield at sub-zero temperatures.",
            patent_id="P1",
            chunk_id=0,
        ),
        _chunk(
            "A catalytic synthesis method achieving high yield at ambient temperature.",
            patent_id="P2",
            chunk_id=0,
        ),
        _chunk(
            "A storage container for chemical reagents.", patent_id="P3", chunk_id=0
        ),
    ]

    res = reranker._score_request_satisfaction(
        parsed,
        "catalytic synthesis high yield sub-zero",
        [0, 1, 2],
        [c.payload["text"] for c in chunks],
        chunks,
    )
    assert res[0][1] == "DIRECT_MATCH"
    assert res[0][0] == 0.92
    assert "functional/constraint" in res[0][2]

    assert res[1][1] == "PARTIAL_MATCH"
    assert res[1][0] == 0.52
    assert "some requested aspects" in res[1][2]

    assert res[2][1] == "NON_MATCH"
    assert res[2][0] == 0.12


def case_25_missing_fields_graceful_handling(_):
    """Missing or empty fields in ParsedQuery must not crash and produce valid [0, 1] satisfaction scores."""
    reranker = _stub_reranker(lambda query, texts: [0.75 for _ in texts])

    # Completely empty parsed query
    parsed_empty = ParsedQuery(original_query="", semantic_query="")
    chunks = [_chunk("Some patent text")]
    res = reranker._score_request_satisfaction(
        parsed_empty, "", [0], ["Some patent text"], chunks
    )
    assert 0 in res
    assert 0.0 <= res[0][0] <= 1.0
    assert res[0][1] in ("DIRECT_MATCH", "PARTIAL_MATCH", "NON_MATCH")

    # None requirements
    res_none = reranker._score_request_satisfaction(
        None, "arbitrary query", [0], ["Some patent text"], chunks
    )
    assert 0 in res_none
    assert 0.0 <= res_none[0][0] <= 1.0


def case_26_contextual_matching_with_section_metadata(_):
    """Candidate context formatting includes section metadata for disambiguation."""
    captured_texts = []

    def fake_score(query, texts):
        captured_texts.extend(texts)
        return [0.88]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        concepts=[
            Concept(id="C1", text="display apparatus", role="object", importance=1.0)
        ]
    )
    chunk = types.SimpleNamespace(
        id="P1-0",
        score=1.0,
        payload={
            "text": "A pixel array circuit",
            "section": "DETAILED DESCRIPTION",
            "patent_id": "P1",
            "chunk_id": 0,
        },
    )
    reranker._score_request_satisfaction(
        parsed, "display apparatus", [0], [chunk.payload["text"]], [chunk]
    )
    assert len(captured_texts) == 1
    assert "Section: DETAILED DESCRIPTION" in captured_texts[0]
    assert "A pixel array circuit" in captured_texts[0]


def case_27_direct_match_outranks_partial_match_when_semantic_comparable(_):
    """Direct match candidate ranks higher than partial match candidate when semantic scores are comparable."""

    def fake_score(query, texts):
        # probe scores: index 0 is DIRECT_MATCH (0.90), index 1 is PARTIAL_MATCH (0.50)
        return [0.90, 0.50]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        concepts=[
            Concept(id="C1", text="battery cathode", role="object", importance=1.0)
        ]
    )
    chunk_direct = _chunk(
        "Battery cathode composition with nickel-manganese", patent_id="P_DIRECT"
    )
    chunk_partial = _chunk(
        "General electrode structure with partial cathode coating",
        patent_id="P_PARTIAL",
    )
    results = [chunk_direct, chunk_partial]

    # Equal coarse semantic scores
    ranked = reranker._blend_and_sort(
        parsed.semantic_query, [0.80, 0.80], results, parsed
    )
    assert ranked[0][1].id == "P_DIRECT-0"
    assert ranked[1][1].id == "P_PARTIAL-0"
    assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
    assert ranked[1][2].request_satisfaction_label == "PARTIAL_MATCH"
    assert ranked[0][0] > ranked[1][0]


def case_28_partial_match_remains_eligible_and_outranks_non_match(_):
    """PARTIAL_MATCH candidate remains eligible and decisively outranks a high-semantic NON_MATCH candidate."""

    def fake_score(query, texts):
        scores = []
        for t in texts:
            if "Photovoltaic" in t:
                scores.append(0.50)  # PARTIAL_MATCH
            else:
                scores.append(0.05)  # NON_MATCH
        return scores

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        concepts=[Concept(id="C1", text="solar cell", role="object", importance=1.0)]
    )
    chunk_partial = _chunk("Photovoltaic solar module component", patent_id="P_PARTIAL")
    chunk_non = _chunk(
        "Solar powered calculator housing with unrelated plastics", patent_id="P_NON"
    )
    results = [chunk_partial, chunk_non]

    # Candidate NON has higher semantic similarity (0.92) than Candidate PARTIAL (0.70)
    ranked = reranker._blend_and_sort(
        parsed.semantic_query, [0.70, 0.92], results, parsed
    )
    assert ranked[0][1].id == "P_PARTIAL-0", (
        f"Expected PARTIAL_MATCH to outrank high-semantic NON_MATCH, got {ranked[0][1].id}"
    )
    assert ranked[1][1].id == "P_NON-0"
    assert ranked[0][0] > ranked[1][0]


def case_29_equivalent_satisfaction_preserves_semantic_order(_):
    """When candidates have equivalent request satisfaction, existing semantic signals determine order."""

    def fake_score(query, texts):
        # Both are direct matches with identical satisfaction score (0.90)
        return [0.90 for _ in texts]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(
        concepts=[
            Concept(id="C1", text="drone navigation", role="object", importance=1.0)
        ]
    )
    chunk_high = _chunk(
        "Autonomous drone navigation using LiDAR and SLAM", patent_id="P_HIGH"
    )
    chunk_mid = _chunk("Drone navigation using basic GPS waypoints", patent_id="P_MID")
    results = [chunk_high, chunk_mid]

    ranked = reranker._blend_and_sort(
        parsed.semantic_query, [0.90, 0.75], results, parsed
    )
    assert ranked[0][1].id == "P_HIGH-0"
    assert ranked[1][1].id == "P_MID-0"
    assert ranked[0][0] > ranked[1][0]


def case_30_filter_non_match_candidates_config(_):
    """When FILTER_NON_MATCH_CANDIDATES is True, NON_MATCH candidates are filtered from final list."""
    import app.reranker as reranker_module

    orig_filter = reranker_module.FILTER_NON_MATCH_CANDIDATES
    reranker_module.FILTER_NON_MATCH_CANDIDATES = True
    try:

        def fake_score(query, texts):
            scores = []
            for t in texts:
                if "fiber laser" in t:
                    scores.append(0.90)  # DIRECT_MATCH
                else:
                    scores.append(0.05)  # NON_MATCH
            return scores

        reranker = _stub_reranker(fake_score)
        parsed = _parsed(
            concepts=[
                Concept(id="C1", text="laser welding", role="object", importance=1.0)
            ]
        )
        chunk_direct = _chunk(
            "High power fiber laser welding system", patent_id="P_DIRECT"
        )
        chunk_non = _chunk("Laser printer toner cartridge assembly", patent_id="P_NON")
        results = [chunk_direct, chunk_non]

        ranked = reranker._blend_and_sort(
            parsed.semantic_query, [0.80, 0.95], results, parsed
        )
        ids = [chunk.id for _, chunk, _ in ranked]
        assert "P_DIRECT-0" in ids
        assert "P_NON-0" not in ids, (
            f"NON_MATCH candidate should have been filtered out, got {ids}"
        )
    finally:
        reranker_module.FILTER_NON_MATCH_CANDIDATES = orig_filter


def case_31_patent_aggregation_prefers_direct_match_representative_chunk(_):
    """Within a patent, chunk with DIRECT_MATCH is selected as best_chunk over high semantic NON_MATCH."""
    from app.semantic_search import SemanticSearch

    # Create mock reranked chunks for single patent P1
    chunk_non = types.SimpleNamespace(
        id="P1-0",
        payload={
            "patent_id": "P1",
            "chunk_id": 0,
            "section": "ABSTRACT",
            "text": "High semantic insect screen",
        },
    )
    breakdown_non = ScoreBreakdown(
        semantic_score=0.92,
        final_score=0.184,
        request_satisfaction_score=0.10,
        request_satisfaction_label="NON_MATCH",
    )

    chunk_direct = types.SimpleNamespace(
        id="P1-1",
        payload={
            "patent_id": "P1",
            "chunk_id": 1,
            "section": "CLAIMS",
            "text": "OLED television display screen",
        },
    )
    breakdown_direct = ScoreBreakdown(
        semantic_score=0.78,
        final_score=0.82,
        request_satisfaction_score=0.90,
        request_satisfaction_label="DIRECT_MATCH",
    )

    reranked = [
        (0.184, chunk_non, breakdown_non),
        (0.82, chunk_direct, breakdown_direct),
    ]

    searcher = SemanticSearch.__new__(SemanticSearch)
    searcher.db = types.SimpleNamespace(
        get_patents_metadata=lambda ids: {pid: {} for pid in ids}
    )

    results = searcher._aggregate_by_patent(reranked)
    assert len(results) == 1
    p = results[0]
    assert p.patent_id == "P1"
    assert p.best_chunk.chunk_id == 1  # The DIRECT_MATCH chunk was chosen as best_chunk
    assert p.request_satisfaction_label == "DIRECT_MATCH"
    assert p.score == 0.82


def case_32_patent_aggregation_direct_beats_high_semantic_non_match_patent(_):
    """Patent with DIRECT_MATCH evidence outranks a patent with only high semantic NON_MATCH evidence."""
    from app.semantic_search import SemanticSearch

    # Patent A: only high semantic NON_MATCH chunks
    chunk_a = types.SimpleNamespace(
        id="PA-0",
        payload={
            "patent_id": "PA",
            "chunk_id": 0,
            "section": "DESC",
            "text": "Irrelevant semantic text",
        },
    )
    breakdown_a = ScoreBreakdown(
        semantic_score=0.95,
        final_score=0.19,
        request_satisfaction_score=0.15,
        request_satisfaction_label="NON_MATCH",
    )

    # Patent B: lower semantic DIRECT_MATCH chunk
    chunk_b = types.SimpleNamespace(
        id="PB-0",
        payload={
            "patent_id": "PB",
            "chunk_id": 0,
            "section": "DESC",
            "text": "Direct match technical text",
        },
    )
    breakdown_b = ScoreBreakdown(
        semantic_score=0.75,
        final_score=0.80,
        request_satisfaction_score=0.88,
        request_satisfaction_label="DIRECT_MATCH",
    )

    reranked = [
        (0.19, chunk_a, breakdown_a),
        (0.80, chunk_b, breakdown_b),
    ]

    searcher = SemanticSearch.__new__(SemanticSearch)
    searcher.db = types.SimpleNamespace(
        get_patents_metadata=lambda ids: {pid: {} for pid in ids}
    )

    results = searcher._aggregate_by_patent(reranked)
    assert len(results) >= 1
    assert results[0].patent_id == "PB"
    assert results[0].request_satisfaction_label == "DIRECT_MATCH"


def case_33_patent_aggregation_partial_match_eligible(_):
    """Patent with PARTIAL_MATCH remains eligible and accessible with its breakdown."""
    from app.semantic_search import SemanticSearch

    chunk = types.SimpleNamespace(
        id="P_PART-0",
        payload={
            "patent_id": "P_PART",
            "chunk_id": 0,
            "section": "DESC",
            "text": "Partial match text",
        },
    )
    breakdown = ScoreBreakdown(
        semantic_score=0.70,
        final_score=0.595,
        request_satisfaction_score=0.50,
        request_satisfaction_label="PARTIAL_MATCH",
        request_satisfaction_reason="Partially satisfies requirements",
    )
    reranked = [(0.595, chunk, breakdown)]

    searcher = SemanticSearch.__new__(SemanticSearch)
    searcher.db = types.SimpleNamespace(
        get_patents_metadata=lambda ids: {pid: {} for pid in ids}
    )

    results = searcher._aggregate_by_patent(reranked)
    assert len(results) == 1
    assert results[0].patent_id == "P_PART"
    assert results[0].request_satisfaction_label == "PARTIAL_MATCH"
    assert results[0].request_satisfaction_score == 0.50
    assert results[0].request_satisfaction_reason == "Partially satisfies requirements"


def case_34_patent_aggregation_multiple_chunks_sorted(_):
    """Multiple chunks of a patent are aggregated and preserved in matching_chunks."""
    from app.semantic_search import SemanticSearch

    chunks = [
        types.SimpleNamespace(
            id="P1-0",
            payload={"patent_id": "P1", "chunk_id": 0, "section": "A", "text": "T0"},
        ),
        types.SimpleNamespace(
            id="P1-1",
            payload={"patent_id": "P1", "chunk_id": 1, "section": "B", "text": "T1"},
        ),
        types.SimpleNamespace(
            id="P1-2",
            payload={"patent_id": "P1", "chunk_id": 2, "section": "C", "text": "T2"},
        ),
    ]
    breakdowns = [
        ScoreBreakdown(
            semantic_score=0.5,
            final_score=0.4,
            request_satisfaction_label="PARTIAL_MATCH",
        ),
        ScoreBreakdown(
            semantic_score=0.8,
            final_score=0.85,
            request_satisfaction_label="DIRECT_MATCH",
        ),
        ScoreBreakdown(
            semantic_score=0.3, final_score=0.2, request_satisfaction_label="NON_MATCH"
        ),
    ]
    reranked = [
        (0.4, chunks[0], breakdowns[0]),
        (0.85, chunks[1], breakdowns[1]),
        (0.2, chunks[2], breakdowns[2]),
    ]
    searcher = SemanticSearch.__new__(SemanticSearch)
    searcher.db = types.SimpleNamespace(
        get_patents_metadata=lambda ids: {pid: {} for pid in ids}
    )

    results = searcher._aggregate_by_patent(reranked)
    assert len(results) == 1
    p = results[0]
    assert p.chunk_count == 3
    assert p.best_chunk.chunk_id == 1


def case_35_unassessed_state_and_safe_fallback(_):
    """UNASSESSED state is safe and model exceptions fallback gracefully without crashing."""
    # 1. Unevaluated ScoreBreakdown defaults to UNASSESSED
    default_bd = ScoreBreakdown()
    assert default_bd.request_satisfaction_label == "UNASSESSED"
    assert default_bd.request_satisfaction_score == 0.0

    # 2. Model failure fallback
    def failing_score(q, texts):
        raise RuntimeError("Reranker model network timeout")

    reranker = _stub_reranker(failing_score)
    parsed = _parsed(
        concepts=[
            Concept(id="C1", text="fiber optics", role="technology", importance=1.0)
        ]
    )
    chunk = _chunk("High bandwidth optical fiber")
    res = reranker._score_request_satisfaction(
        parsed, "fiber optics", [0], [chunk.payload["text"]], [chunk]
    )
    assert res[0][1] == "NON_MATCH"
    assert res[0][0] == 0.0

    # 3. Non-fine candidates in _blend_and_sort have UNASSESSED
    import app.reranker as reranker_module

    orig_top_n = reranker_module.RERANK_FINE_STAGE_TOP_N
    reranker_module.RERANK_FINE_STAGE_TOP_N = 1
    try:
        ok_reranker = _stub_reranker(lambda q, texts: [0.90])
        chunks = [
            _chunk("Top fine chunk", chunk_id=0),
            _chunk("Coarse candidate", chunk_id=1),
        ]
        ranked = ok_reranker._blend_and_sort(
            "fiber optics", [0.95, 0.70], chunks, parsed
        )
        assert ranked[0][2].request_satisfaction_label == "DIRECT_MATCH"
        assert ranked[1][2].request_satisfaction_label == "UNASSESSED"
        assert ranked[1][2].request_satisfaction_score == 0.0
    finally:
        reranker_module.RERANK_FINE_STAGE_TOP_N = orig_top_n


CASES = [
    (
        "Test 1  - simple query weights favor semantic",
        case_1_simple_query_weights_favor_semantic,
    ),
    (
        "Test 2  - optimization-only query keeps semantic floor",
        case_2_optimization_only_query_does_not_break_floor,
    ),
    ("Test 3  - relationship weight is clamped", case_3_relationship_weight_is_clamped),
    (
        "Test 4  - LLM zero-semantic-weight is ignored",
        case_4_llm_zero_semantic_weight_is_ignored,
    ),
    ("Test 5  - weights always sum to one", case_5_weights_always_sum_to_one),
    (
        "Test 6  - has_requirements_structure empty is false",
        case_6_has_requirements_structure_empty_is_false,
    ),
    (
        "Test 7  - has_requirements_structure optimization-only is true",
        case_7_has_requirements_structure_optimization_only_is_true,
    ),
    (
        "Test 8  - optimization gating suppresses off-topic noise",
        case_8_optimization_gating_suppresses_offtopic_noise,
    ),
    (
        "Test 9  - optimization gating rewards aligned direction",
        case_9_optimization_gating_rewards_aligned_direction,
    ),
    (
        "Test 10 - optimization gating penalizes opposed direction",
        case_10_optimization_gating_penalizes_opposed_direction,
    ),
    (
        "Test 11 - exclusion literal hit outweighs semantic-only",
        case_11_exclusion_literal_hit_outweighs_semantic_only,
    ),
    (
        "Test 12 - exclusion semantic paraphrase still detected",
        case_12_exclusion_semantic_paraphrase_still_detected,
    ),
    (
        "Test 13 - relationship scoring is importance-weighted",
        case_13_relationship_scoring_is_importance_weighted,
    ),
    (
        "Test 14 - pure fallback matches semantic order",
        case_14_pure_fallback_matches_semantic_order,
    ),
    (
        "Test 15 - non-fine-stage chunks keep pure semantic score",
        case_15_non_fine_stage_chunks_keep_pure_semantic_score,
    ),
    (
        "Test 16 - defect detection (relationship)",
        case_16_defect_detection_relationship_query,
    ),
    (
        "Test 17 - power consumption (optimization: minimize)",
        case_17_power_consumption_minimize_query,
    ),
    ("Test 18 - plant meat taste (goal, paraphrase)", case_18_plant_meat_taste_query),
    (
        "Test 19 - sweetness + low sugar (goal + constraint)",
        case_19_sweetness_sugar_constraint_query,
    ),
    (
        "Test 20 - vacuum pressure machine (dual optimization)",
        case_20_vacuum_pressure_dual_optimization_query,
    ),
    (
        "Test 21 - request satisfaction labels and breakdown",
        case_21_request_satisfaction_labels_and_breakdown,
    ),
    (
        "Test 22 - non-match penalty overrides higher semantic score",
        case_22_request_satisfaction_penalizes_non_match_over_lower_semantic_direct,
    ),
    (
        "Test 23 - question query target satisfaction",
        case_23_question_query_satisfaction_evaluates_target,
    ),
    (
        "Test 24 - multi-part request satisfaction",
        case_24_multipart_request_satisfaction,
    ),
    (
        "Test 25 - missing fields graceful handling",
        case_25_missing_fields_graceful_handling,
    ),
    (
        "Test 26 - contextual matching with section metadata",
        case_26_contextual_matching_with_section_metadata,
    ),
    (
        "Test 27 - direct match outranks partial match when comparable",
        case_27_direct_match_outranks_partial_match_when_semantic_comparable,
    ),
    (
        "Test 28 - partial match remains eligible and outranks non-match",
        case_28_partial_match_remains_eligible_and_outranks_non_match,
    ),
    (
        "Test 29 - equivalent satisfaction preserves semantic order",
        case_29_equivalent_satisfaction_preserves_semantic_order,
    ),
    (
        "Test 30 - filter non-match candidates config",
        case_30_filter_non_match_candidates_config,
    ),
    (
        "Test 31 - patent aggregation prefers direct match chunk",
        case_31_patent_aggregation_prefers_direct_match_representative_chunk,
    ),
    (
        "Test 32 - patent aggregation direct beats high-semantic non-match",
        case_32_patent_aggregation_direct_beats_high_semantic_non_match_patent,
    ),
    (
        "Test 33 - patent aggregation partial match eligible",
        case_33_patent_aggregation_partial_match_eligible,
    ),
    (
        "Test 34 - patent aggregation multiple chunks sorted",
        case_34_patent_aggregation_multiple_chunks_sorted,
    ),
    (
        "Test 35 - unassessed state and safe fallback",
        case_35_unassessed_state_and_safe_fallback,
    ),
]


def main():
    failures = []
    for name, case in CASES:
        try:
            case(None)
            print(f"[PASS] {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failures.append(name)
            print(f"[ERROR] {name}: {e!r}")

    print()
    if failures:
        print(f"{len(failures)} of {len(CASES)} cases FAILED: {failures}")
        sys.exit(1)
    else:
        print(
            f"All {len(CASES)} cases passed (or gracefully skipped where the local model is unavailable)."
        )


if __name__ == "__main__":
    main()

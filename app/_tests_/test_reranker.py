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
    parsed = _parsed(concepts=[Concept(id="C1", text="bottle design", role="object", importance=0.8)])
    w = _compute_weights(parsed)
    assert w["semantic"] + w["structured"] >= 0.55
    assert w["relationship"] == 0.0
    assert w["optimization"] == 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-9


def case_2_optimization_only_query_does_not_break_floor(_):
    """A pure-optimization query (no Concept/Goal/Constraint at all) still keeps the semantic floor."""
    parsed = _parsed(optimization=[OptimizationTarget(id="O1", property="power consumption", direction="minimize", importance=0.9)])
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
        relationships=[Relationship(source="camera", relation="used_for", target="defect detection", importance=1.0)],
        ranking_weights=RankingWeights(semantic_relevance=1.0, relationship_satisfaction=0.9),
    )
    w = _compute_weights(parsed)
    assert w["relationship"] <= 0.20 + 1e-9
    assert w["semantic"] + w["structured"] >= 0.55


def case_4_llm_zero_semantic_weight_is_ignored(_):
    """Even if the LLM outputs semantic_relevance=0.0, application code keeps semantic dominant."""
    parsed = _parsed(
        concepts=[Concept(id="C1", text="widget", role="object", importance=0.9)],
        relationships=[Relationship(source="a", relation="controls", target="b", importance=1.0)],
        ranking_weights=RankingWeights(semantic_relevance=0.0, relationship_satisfaction=1.0),
    )
    w = _compute_weights(parsed)
    assert w["semantic"] + w["structured"] >= 0.55, "LLM's semantic_relevance=0.0 must not be trusted raw"


def case_5_weights_always_sum_to_one(_):
    scenarios = [
        _parsed(),
        _parsed(concepts=[Concept(id="C1", text="x", role="object", importance=1.0)]),
        _parsed(
            concepts=[Concept(id="C1", text="x", role="object", importance=1.0)],
            goals=[Goal(id="G1", text="y", importance=1.0, required=True)],
            constraints=[Constraint(id="K1", text="z", importance=1.0, required=True)],
            optimization=[OptimizationTarget(id="O1", property="p", direction="maximize", importance=1.0)],
            relationships=[Relationship(source="a", relation="r", target="b", importance=1.0)],
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
    parsed = _parsed(optimization=[OptimizationTarget(id="O1", property="cost", direction="minimize")])
    assert parsed.has_requirements_structure is True


# ==================================================================
# Tier 1 - stubbed-model gating/blend logic
# ==================================================================


def case_8_optimization_gating_suppresses_offtopic_noise(_):
    """
    A chunk that doesn't discuss the property at all should land near
    neutral (0.5), not get pulled toward either direction by noise.
    """
    reranker = _stub_reranker(lambda query, texts: [0.05 for _ in texts])  # both aligned & opposed score low
    target = OptimizationTarget(id="O1", property="power consumption", direction="minimize", importance=1.0)
    scores = reranker._score_optimization([target], fine_indices=[0], fine_texts=["irrelevant text"])
    assert abs(scores[0] - 0.5) < 0.05, scores


def case_9_optimization_gating_rewards_aligned_direction(_):
    """A chunk clearly discussing the aligned direction should score above neutral."""

    def fake_score(query, texts):
        # "reduces"/"low" phrasing scores high, "increases"/"high" phrasing scores low
        if "reduces" in query or "low" in query:
            return [0.9 for _ in texts]
        return [0.1 for _ in texts]

    reranker = _stub_reranker(fake_score)
    target = OptimizationTarget(id="O1", property="power consumption", direction="minimize", importance=1.0)
    scores = reranker._score_optimization([target], fine_indices=[0], fine_texts=["reduces power consumption"])
    assert scores[0] > 0.5, scores


def case_10_optimization_gating_penalizes_opposed_direction(_):
    def fake_score(query, texts):
        if "increases" in query or "high" in query:
            return [0.9 for _ in texts]
        return [0.1 for _ in texts]

    reranker = _stub_reranker(fake_score)
    target = OptimizationTarget(id="O1", property="power consumption", direction="minimize", importance=1.0)
    scores = reranker._score_optimization([target], fine_indices=[0], fine_texts=["increases power consumption"])
    assert scores[0] < 0.5, scores


def case_11_exclusion_literal_hit_outweighs_semantic_only(_):
    """A literal verbatim exclusion match must be treated as at least as strong as any semantic-only signal."""
    reranker = _stub_reranker(lambda query, texts: [0.2 for _ in texts])  # low semantic signal
    results = [_chunk("this uses lithium directly")]
    penalties = reranker._score_exclusions(["lithium"], fine_indices=[0], fine_texts=["this uses lithium directly"], results=results)
    assert penalties[0] == 1.0, penalties


def case_12_exclusion_semantic_paraphrase_still_detected(_):
    """A paraphrased exclusion mention (no literal term) is still caught via the semantic probe."""
    reranker = _stub_reranker(lambda query, texts: [0.85 for _ in texts])  # high semantic signal, no literal term
    results = [_chunk("the cell chemistry relies on an alkali-metal-ion compound")]
    penalties = reranker._score_exclusions(["lithium"], fine_indices=[0], fine_texts=["..."], results=results)
    assert penalties[0] == 0.85, penalties


def case_13_relationship_scoring_is_importance_weighted(_):
    # _build_relationship_sentence renders "{source} {relation} {target}" -
    # match on the relation text embedded in that templated sentence.
    def fake_score(query, texts):
        return [1.0 if "important relation" in query else 0.0 for _ in texts]

    reranker = _stub_reranker(fake_score)
    relationships = [
        Relationship(source="a", relation="important relation", target="b", importance=1.0),
        Relationship(source="c", relation="minor relation", target="d", importance=0.1),
    ]
    scores = reranker._score_relationships(relationships, fine_indices=[0], fine_texts=["text"])
    assert scores[0] > 0.85, scores  # dominated by the high-importance relationship


def case_14_pure_fallback_matches_semantic_order(_):
    """No requirements structure -> rerank() falls back to a pure semantic-score sort."""
    reranker = _stub_reranker(lambda query, texts: [0.2, 0.9, 0.5])
    results = [_chunk("a", chunk_id=0), _chunk("b", chunk_id=1), _chunk("c", chunk_id=2)]
    ranked = reranker.rerank("query", results, requirements=None)
    assert [r[1].id for r in ranked] == [results[1].id, results[2].id, results[0].id]
    assert all(r[2].final_score == r[0] for r in ranked)  # ScoreBreakdown always populated


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
        parsed = _parsed(concepts=[Concept(id="C1", text="on-target", role="object", importance=0.9)])

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


def _assert_on_target_wins(case_name, parsed, on_target_text, off_target_text, on_target_semantic, off_target_semantic):
    reranker = _get_real_reranker()
    if reranker is None:
        print(f"[SKIP] {case_name} - local reranker model unavailable: {_REAL_RERANKER_ERROR}")
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
        ranked = reranker._blend_and_sort(parsed.semantic_query, [on_target_semantic, off_target_semantic], results, parsed)
    finally:
        reranker._score = real_score

    order = [chunk.id for _, chunk, _ in ranked]
    assert order[0] == "ON-0", f"{case_name}: expected on-target chunk to win, got order {order}"
    print(f"[OK]   {case_name} -> on-target chunk correctly outranked off-target chunk")


def case_16_defect_detection_relationship_query(_):
    parsed = _parsed(
        semantic_query="automatically detecting defects in manufactured products using cameras and AI",
        concepts=[
            Concept(id="C1", text="manufactured products", role="object", importance=0.7),
            Concept(id="C2", text="camera", role="technology", importance=0.8, semantic_variants=["imaging sensor"]),
            Concept(id="C3", text="AI", role="technology", importance=0.8, semantic_variants=["neural network"]),
        ],
        goals=[Goal(id="G1", text="automatically detect defects", importance=0.9, required=True, keywords=["identifies defective articles", "detects defects"])],
        relationships=[Relationship(source="camera", relation="used_for", target="defect detection", importance=0.9)],
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
        concepts=[Concept(id="C1", text="semiconductor devices", role="object", importance=0.8)],
        optimization=[OptimizationTarget(id="O1", property="power consumption", direction="minimize", importance=0.9)],
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
        concepts=[Concept(id="C1", text="plant-based meat", role="object", importance=0.9, semantic_variants=["meat analogue", "meat substitute"])],
        goals=[Goal(id="G1", text="improve taste", importance=0.9, required=True, keywords=["improved palatability", "enhanced flavor profile"])],
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
        goals=[Goal(id="G1", text="increase sweetness", importance=0.9, required=True, keywords=["enhanced sweetness", "sweetening effect"])],
        constraints=[Constraint(id="K1", text="keep sugar low", importance=0.9, required=True, keywords=["reduced sugar content", "low-sugar"])],
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
        concepts=[Concept(id="C1", text="vacuum pressure machine", role="object", importance=0.9)],
        optimization=[
            OptimizationTarget(id="O1", property="electricity consumption", direction="minimize", importance=0.8),
            OptimizationTarget(id="O2", property="performance", direction="maximize", importance=0.8),
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


CASES = [
    ("Test 1  - simple query weights favor semantic", case_1_simple_query_weights_favor_semantic),
    ("Test 2  - optimization-only query keeps semantic floor", case_2_optimization_only_query_does_not_break_floor),
    ("Test 3  - relationship weight is clamped", case_3_relationship_weight_is_clamped),
    ("Test 4  - LLM zero-semantic-weight is ignored", case_4_llm_zero_semantic_weight_is_ignored),
    ("Test 5  - weights always sum to one", case_5_weights_always_sum_to_one),
    ("Test 6  - has_requirements_structure empty is false", case_6_has_requirements_structure_empty_is_false),
    ("Test 7  - has_requirements_structure optimization-only is true", case_7_has_requirements_structure_optimization_only_is_true),
    ("Test 8  - optimization gating suppresses off-topic noise", case_8_optimization_gating_suppresses_offtopic_noise),
    ("Test 9  - optimization gating rewards aligned direction", case_9_optimization_gating_rewards_aligned_direction),
    ("Test 10 - optimization gating penalizes opposed direction", case_10_optimization_gating_penalizes_opposed_direction),
    ("Test 11 - exclusion literal hit outweighs semantic-only", case_11_exclusion_literal_hit_outweighs_semantic_only),
    ("Test 12 - exclusion semantic paraphrase still detected", case_12_exclusion_semantic_paraphrase_still_detected),
    ("Test 13 - relationship scoring is importance-weighted", case_13_relationship_scoring_is_importance_weighted),
    ("Test 14 - pure fallback matches semantic order", case_14_pure_fallback_matches_semantic_order),
    ("Test 15 - non-fine-stage chunks keep pure semantic score", case_15_non_fine_stage_chunks_keep_pure_semantic_score),
    ("Test 16 - defect detection (relationship)", case_16_defect_detection_relationship_query),
    ("Test 17 - power consumption (optimization: minimize)", case_17_power_consumption_minimize_query),
    ("Test 18 - plant meat taste (goal, paraphrase)", case_18_plant_meat_taste_query),
    ("Test 19 - sweetness + low sugar (goal + constraint)", case_19_sweetness_sugar_constraint_query),
    ("Test 20 - vacuum pressure machine (dual optimization)", case_20_vacuum_pressure_dual_optimization_query),
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
        print(f"All {len(CASES)} cases passed (or gracefully skipped where the local model is unavailable).")


if __name__ == "__main__":
    main()

"""
Reranker

Choose between a remote reranking server and a local sentence-transformers
CrossEncoder using a single config flag.

Design (see /Users/hepinprolix/.claude/plans/snazzy-drifting-owl.md):

When a ParsedQuery with a requirements structure (concepts, goals,
constraints, optimization, exclusions, relationships, requirements,
ranking_weights - see app/query_understanding/models.py) is supplied,
the reranker reuses the SAME already-loaded model (local CrossEncoder
or remote endpoint) as a general-purpose (query-like-text, passage)
relevance scorer for a small, bounded set of additional
deterministically-templated synthetic queries built from that
structure - not literal substring matching, and not another LLM call.
Every extra query is scored once per user query, batched over the same
candidate-chunk set already fetched, so cost scales with query
structural complexity (how many relationships/optimization targets/
exclusions were extracted - typically <=10 total), never with chunk
count or technology domain.

All model scores are 0-1 relevance scores, not statistical
probabilities - sigmoid bounding (local) or a server-provided
relevance score (remote) establishes a 0-1 range, not statistical
calibration. Locally, CrossEncoder's activation_fn sigmoid-bounds the
raw logit into 0-1. Remotely, the endpoint's own relevance_score is
used as-is under a documented [0,1] assumption (validated, not
rescaled - see _score()). Neither path uses batch-relative min-max
normalization (that was the prior design's flaw: it destroys the
absolute meaning of a score by force-stretching whatever the batch's
best candidate happens to be to 1.0, even when every candidate is
mediocre).

Without a requirements structure (or when it's empty), reranking is
pure semantic score, exactly as before this redesign.
"""

from __future__ import annotations

import requests
import torch
from sentence_transformers import CrossEncoder

from app.config import (
    EXCLUSION_HARD_FILTER_THRESHOLD,
    EXCLUSION_PENALTY_WEIGHT,
    LOCAL_RERANKER_MODEL,
    MAX_EXCLUSIONS_SCORED,
    MAX_OPTIMIZATION_TARGETS_SCORED,
    MAX_RELATIONSHIPS_SCORED,
    MAX_SECONDARY_WEIGHT,
    MAX_WEAK_SIGNAL_WEIGHT,
    MIN_SEMANTIC_WEIGHT,
    REQUIRED_IMPORTANCE_BOOST,
    RERANK_FINE_STAGE_TOP_N,
    RERANKER_REMOTE_API_KEY,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RERANKER_REQUEST_TIMEOUT,
    SEM_STRUCT_SPLIT_RATIO,
    STRUCTURE_COVERAGE_MIN_MULTIPLIER,
    STRUCTURED_SIGNAL_MIN_IMPORTANCE,
    USE_REMOTE_RERANKER,
)
from app.models.patent_search_result import ScoreBreakdown
from app.query_understanding.models import ParsedQuery


def _weight(importance: float, required: bool) -> float:
    return min(1.0, importance + (REQUIRED_IMPORTANCE_BOOST if required else 0.0))


# ==================================================================
# Lexical coverage - kept as a small "weak supporting" signal only
# (see MAX_WEAK_SIGNAL_WEIGHT). Literal overlap is a real but weak
# clue for patent text, never the primary evidence - the semantic
# signals below (structured/relationship/optimization scores) are
# what actually replace substring matching as the primary mechanism.
# ==================================================================


def _weighted_term_coverage(
    text_lower: str, terms: list[tuple[list[str], float]]
) -> float:
    if not terms:
        return 0.0

    total = sum(weight for _, weight in terms) or 1.0
    covered = sum(
        weight
        for surface_forms, weight in terms
        if any(sf and sf.lower() in text_lower for sf in surface_forms)
    )
    return covered / total


def _concept_terms(requirements: ParsedQuery) -> list[tuple[list[str], float]]:
    terms = [
        ([c.text, *c.semantic_variants], _weight(c.importance, c.required))
        for c in requirements.concepts
    ]
    terms += [
        ([g.text, *g.keywords], _weight(g.importance, g.required))
        for g in requirements.goals
    ]
    terms += [
        ([o.property], _weight(o.importance, False)) for o in requirements.optimization
    ]
    terms += [
        ([r.description, *r.keywords], _weight(r.importance, r.required))
        for r in requirements.requirements
        if not _is_constraint_type(r.type)
    ]
    return terms


def _is_constraint_type(requirement_type: str) -> bool:
    t = (requirement_type or "").lower()
    return "constraint" in t or "exclusion" in t


def _constraint_coverage(text_lower: str, requirements: ParsedQuery) -> float:
    scores = []

    constraint_terms = [
        ([k.text, *k.keywords], _weight(k.importance, k.required))
        for k in requirements.constraints
    ]
    constraint_terms += [
        ([r.description, *r.keywords], _weight(r.importance, r.required))
        for r in requirements.requirements
        if _is_constraint_type(r.type)
    ]
    if constraint_terms:
        scores.append(_weighted_term_coverage(text_lower, constraint_terms))

    if requirements.exclusions:
        hits = sum(1 for term in requirements.exclusions if term.lower() in text_lower)
        scores.append(1.0 - hits / len(requirements.exclusions))

    return sum(scores) / len(scores) if scores else 0.0


def _lexical_score(text_lower: str, requirements: ParsedQuery) -> float:
    parts = []
    concept_terms = _concept_terms(requirements)
    if concept_terms:
        parts.append(_weighted_term_coverage(text_lower, concept_terms))
    if requirements.constraints or requirements.exclusions:
        parts.append(_constraint_coverage(text_lower, requirements))
    return sum(parts) / len(parts) if parts else 0.0


def _exact_match_score(text_lower: str, query: str) -> float:
    q = query.strip().lower()
    return 1.0 if q and q in text_lower else 0.0


# ==================================================================
# Deterministic synthetic-query template builders. Built in plain
# Python from ParsedQuery fields the Query Understanding LLM already
# extracted once, at query time - no LLM call happens here.
# ==================================================================


def _build_structured_sentence(requirements: ParsedQuery) -> str | None:
    """
    ONE natural-language sentence assembled from the required/important
    concepts, goals, and constraints - scored via the same reranker
    model as a secondary semantic signal, so paraphrased patent
    language sharing no literal tokens with the user's wording can
    still be recognized as satisfying the extracted requirements.
    """

    concept_bits = [
        c.text
        for c in requirements.concepts
        if c.required or c.importance >= STRUCTURED_SIGNAL_MIN_IMPORTANCE
    ]
    goal_bits = [
        g.text
        for g in requirements.goals
        if g.required or g.importance >= STRUCTURED_SIGNAL_MIN_IMPORTANCE
    ]
    constraint_bits = [
        k.text
        for k in requirements.constraints
        if k.required or k.importance >= STRUCTURED_SIGNAL_MIN_IMPORTANCE
    ]

    if not (concept_bits or goal_bits or constraint_bits):
        return None

    parts = []
    if concept_bits:
        parts.append(", ".join(concept_bits))
    if goal_bits:
        parts.append("achieving " + ", ".join(goal_bits))
    if constraint_bits:
        parts.append("while " + ", ".join(constraint_bits))

    return "; ".join(parts)


def _build_relationship_sentence(relationship) -> str:
    verb = relationship.relation.replace("_", " ").strip() or "relates to"
    return f"{relationship.source} {verb} {relationship.target}"


def _build_optimization_pair(target) -> tuple[str, str]:
    """
    (aligned, opposed) phrasing pair for a directional optimization
    target - e.g. minimize "power consumption" -> aligned talks about
    reducing/low power consumption, opposed about increasing/high.
    """

    if target.direction == "minimize":
        aligned = f"reduces {target.property}, low {target.property}"
        opposed = f"increases {target.property}, high {target.property}"
    else:
        aligned = f"increases {target.property}, high {target.property}"
        opposed = f"reduces {target.property}, low {target.property}"
    return aligned, opposed


def _build_exclusion_probe(term: str) -> str:
    return f"the invention uses, includes, or involves {term}"


# ==================================================================
# Bounded, application-enforced weight computation. The LLM's
# ranking_weights is read only where a dedicated field exists
# (relationship_satisfaction) and only ever bounded/clamped - it never
# sets the semantic/structured split, and categories with no extracted
# data always get 0 (folded back into semantic), regardless of what
# the LLM output. This is what actually keeps semantic relevance
# dominant, not a courtesy clamp on top of an otherwise-trusted value.
# ==================================================================


def _compute_weights(requirements: ParsedQuery) -> dict:
    llm = requirements.ranking_weights

    has_struct = _build_structured_sentence(requirements) is not None
    has_rel = bool(requirements.relationships)
    has_opt = bool(requirements.optimization)
    has_lex = (
        bool(_concept_terms(requirements))
        or bool(requirements.constraints)
        or bool(requirements.exclusions)
    )

    # Step 1/2: clamp per-category weight. relationship_satisfaction is
    # the only RankingWeights field with a direct new-category mapping;
    # optimization/lexical have no dedicated LLM field in the current
    # schema (no schema change per the approved plan), so they get a
    # fixed ceiling value when the category has data, same spirit as
    # "a small fixed ceiling" already specified for lexical/exact.
    w_rel = (
        min(max(llm.relationship_satisfaction, 0.0), MAX_SECONDARY_WEIGHT)
        if has_rel
        else 0.0
    )
    w_opt = MAX_SECONDARY_WEIGHT if has_opt else 0.0
    w_lex = MAX_WEAK_SIGNAL_WEIGHT if has_lex else 0.0
    w_exact = MAX_WEAK_SIGNAL_WEIGHT

    # Step 3: shrink the clamped remainder further if needed so
    # semantic+structured never drops below its floor.
    remainder = w_rel + w_opt + w_lex + w_exact
    sem_and_struct = 1.0 - remainder
    if sem_and_struct < MIN_SEMANTIC_WEIGHT and remainder > 0:
        scale = (1.0 - MIN_SEMANTIC_WEIGHT) / remainder
        w_rel *= scale
        w_opt *= scale
        w_lex *= scale
        w_exact *= scale
        sem_and_struct = MIN_SEMANTIC_WEIGHT

    # Step 4: fixed (non-LLM-controlled) split of the combined budget.
    if has_struct:
        w_sem = sem_and_struct * SEM_STRUCT_SPLIT_RATIO
        w_struct = sem_and_struct * (1 - SEM_STRUCT_SPLIT_RATIO)
    else:
        w_sem = sem_and_struct
        w_struct = 0.0

    weights = {
        "semantic": w_sem,
        "structured": w_struct,
        "relationship": w_rel,
        "optimization": w_opt,
        "lexical": w_lex,
        "exact": w_exact,
    }

    # Step 5: defensive renormalization (should already sum to ~1.0).
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _validate_remote_score(value: float) -> float:
    """
    Guard against a remote relevance_score that violates the
    documented [0, 1] assumption this blend relies on (the same range
    the local sigmoid-bounded path produces). This is a defensive
    clamp for an out-of-contract value, not a rescale: an in-range
    score is returned completely unchanged. This is deliberately NOT
    batch-relative and NOT a general calibration - if the remote
    endpoint is later confirmed to use a different documented scale,
    follow that scale explicitly instead of adjusting this guard.
    """

    if 0.0 <= value <= 1.0:
        return value

    clamped = min(1.0, max(0.0, value))
    print(
        f"[Reranker] Warning: remote relevance_score {value} is outside the "
        f"expected [0, 1] range - clamping to {clamped}."
    )
    return clamped


class Reranker:
    """
    Rerank Qdrant search results using either the remote server or a local model.
    """

    def __init__(self, use_remote: bool | None = None):
        self.use_remote = USE_REMOTE_RERANKER if use_remote is None else use_remote

        if self.use_remote:
            self.base_url = RERANKER_REMOTE_BASE_URL.rstrip("/")
            self.model = RERANKER_REMOTE_MODEL
            self.headers = {"Authorization": f"Bearer {RERANKER_REMOTE_API_KEY}"}
            self.cross_encoder = None
        else:
            self.cross_encoder = CrossEncoder(LOCAL_RERANKER_MODEL)
            self.base_url = None
            self.model = LOCAL_RERANKER_MODEL
            self.headers = {}

    def rerank(
        self,
        query: str,
        results: list,
        requirements: ParsedQuery | None = None,
    ) -> list[tuple[float, object, ScoreBreakdown]]:
        """
        Rerank Qdrant search results.

        *requirements*, when given a ParsedQuery with a non-empty
        requirements structure, blends the semantic score with
        requirement/relationship/optimization/exclusion evaluation
        (see module docstring). Otherwise ranking is pure semantic
        score.

        Returns every result as a ``(final_score, qdrant_result,
        ScoreBreakdown)`` tuple, sorted by score descending.
        """

        if not results:
            return []

        texts = [result.payload["text"] for result in results]
        print(f"Reranking {len(texts)} candidate chunks for query: {query}")

        semantic_scores = self._score(query, texts)

        if requirements is None or not requirements.has_requirements_structure:
            ranked = sorted(
                zip(semantic_scores, results), key=lambda pair: pair[0], reverse=True
            )
            return [
                (
                    float(score),
                    chunk,
                    ScoreBreakdown(
                        semantic_score=float(score), final_score=float(score)
                    ),
                )
                for score, chunk in ranked
            ]

        return self._blend_and_sort(query, semantic_scores, results, requirements)

    def _score(self, query: str, texts: list[str]) -> list[float]:
        """
        Raw per-chunk relevance scores, aligned index-for-index with
        *texts*. Reused for ANY query string - the original
        semantic_query, or any deterministically-templated synthetic
        query - not just the user's literal query.
        """

        if self.use_remote:
            response = requests.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                },
                timeout=RERANKER_REQUEST_TIMEOUT,
            )
            print(f"Reranker response status code: {response.status_code}")
            response.raise_for_status()

            # The server's relevance_score is used as-is, under the
            # documented assumption (this endpoint's own example
            # response shape, [0, 1]-scaled) that it's already a
            # server-provided relevance score comparable to the local
            # path's sigmoid-bounded one - not rescaled, not
            # batch-normalized. _validate_remote_score only guards
            # against a value that violates that assumption outright;
            # it never transforms an in-range score.
            scores = [0.0] * len(texts)
            for item in response.json()["results"]:
                scores[item["index"]] = _validate_remote_score(
                    float(item["relevance_score"])
                )
            return scores

        # Local CrossEncoder returns unbounded logits by default -
        # activation_fn sigmoid-bounds them into a 0-1 relevance score
        # (bounding alone, not statistical calibration), replacing the
        # prior design's batch-relative min-max normalization (which
        # distorted absolute meaning: a batch of uniformly mediocre
        # candidates would still get force-stretched to a normalized
        # 1.0 for its "least bad" entry).
        pairs = [(query, text) for text in texts]
        scores = self.cross_encoder.predict(
            pairs, show_progress_bar=False, activation_fn=torch.nn.Sigmoid()
        )
        return [float(s) for s in scores]

    # ==============================================================
    # Fine-stage synthetic-query scoring (relationships/optimization/
    # exclusions) - each helper issues ONE additional _score() call per
    # item (two for optimization, since it needs an aligned/opposed
    # pair), batched over the fine-stage chunk subset only.
    # ==============================================================

    def _score_relationships(
        self, relationships: list, fine_indices: list[int], fine_texts: list[str]
    ) -> dict[int, float]:
        if not relationships or not fine_texts:
            return {}

        per_relationship = [
            (rel.importance, self._score(_build_relationship_sentence(rel), fine_texts))
            for rel in relationships
        ]
        total_weight = sum(w for w, _ in per_relationship) or 1.0

        return {
            idx: sum(w * scores[pos] for w, scores in per_relationship) / total_weight
            for pos, idx in enumerate(fine_indices)
        }

    def _score_optimization(
        self, targets: list, fine_indices: list[int], fine_texts: list[str]
    ) -> dict[int, float]:
        if not targets or not fine_texts:
            return {}

        per_target = []
        for target in targets:
            aligned, opposed = _build_optimization_pair(target)
            per_target.append(
                (
                    target.importance,
                    self._score(aligned, fine_texts),
                    self._score(opposed, fine_texts),
                )
            )
        total_weight = sum(w for w, _, _ in per_target) or 1.0

        result = {}
        for pos, idx in enumerate(fine_indices):
            acc = 0.0
            for weight, aligned_scores, opposed_scores in per_target:
                # Property-relevance gating: if neither phrasing is
                # relevant to this chunk (property never discussed),
                # `gate` is low and the directional signal is
                # suppressed toward neutral (0.5) rather than favoring
                # either direction based on noise.
                gate = max(aligned_scores[pos], opposed_scores[pos])
                directional = aligned_scores[pos] - opposed_scores[pos]
                acc += weight * (0.5 + gate * directional / 2)
            result[idx] = acc / total_weight
        return result

    def _score_exclusions(
        self,
        exclusions: list[str],
        fine_indices: list[int],
        fine_texts: list[str],
        results: list,
    ) -> dict[int, float]:
        if not exclusions:
            return {}

        per_term = [
            (
                term,
                self._score(_build_exclusion_probe(term), fine_texts)
                if fine_texts
                else [],
            )
            for term in exclusions
        ]

        result = {}
        for pos, idx in enumerate(fine_indices):
            text_lower = ((results[idx].payload or {}).get("text") or "").lower()
            worst = 0.0
            for term, scores in per_term:
                literal_hit = term.lower() in text_lower
                semantic_value = scores[pos] if scores else 0.0
                # A literal verbatim match is always treated as at
                # least as strong as a semantic-only match (it's forced
                # to the maximum value, 1.0) - literal evidence is
                # stronger than semantic-only evidence, though it's
                # still a SOFT penalty (see EXCLUSION_PENALTY_WEIGHT),
                # since a literal hit could be a negation ("free of
                # lithium") that a substring check can't distinguish.
                value = 1.0 if literal_hit else semantic_value
                worst = max(worst, value)
            result[idx] = worst
        return result

    def _blend_and_sort(
        self,
        query: str,
        semantic_scores: list[float],
        results: list,
        requirements: ParsedQuery,
    ) -> list[tuple[float, object, ScoreBreakdown]]:
        # Two-stage coarse-then-fine: only the top RERANK_FINE_STAGE_TOP_N
        # coarse-semantic-scored chunks get the expensive synthetic-query
        # evaluation. Everything else keeps its coarse semantic score
        # untouched - no arbitrary scaling or penalty is applied to them;
        # a fine-scored chunk only overtakes them by actually earning a
        # higher deterministic final score.
        fine_indices = sorted(
            range(len(results)), key=lambda i: semantic_scores[i], reverse=True
        )[:RERANK_FINE_STAGE_TOP_N]
        fine_set = set(fine_indices)
        fine_texts = [results[i].payload["text"] for i in fine_indices]

        structured_sentence = _build_structured_sentence(requirements)
        structured_scores = (
            dict(zip(fine_indices, self._score(structured_sentence, fine_texts)))
            if structured_sentence
            else {}
        )

        relationships = requirements.relationships[:MAX_RELATIONSHIPS_SCORED]
        relationship_scores = self._score_relationships(
            relationships, fine_indices, fine_texts
        )

        optimization_targets = requirements.optimization[
            :MAX_OPTIMIZATION_TARGETS_SCORED
        ]
        optimization_scores = self._score_optimization(
            optimization_targets, fine_indices, fine_texts
        )

        exclusions = requirements.exclusions[:MAX_EXCLUSIONS_SCORED]
        exclusion_scores = self._score_exclusions(
            exclusions, fine_indices, fine_texts, results
        )

        weights = _compute_weights(requirements)

        # has_struct/has_rel gate the structural-coverage penalty below -
        # both are already known from the variables just computed above,
        # no extra work.
        has_struct = structured_sentence is not None
        has_rel = bool(relationships)

        blended: list[tuple[float, object, ScoreBreakdown]] = []
        for i, chunk in enumerate(results):
            if i in fine_set:
                text_lower = ((chunk.payload or {}).get("text") or "").lower()
                sem = semantic_scores[i]
                struct = structured_scores.get(i, 0.0)
                rel = relationship_scores.get(i, 0.0)
                opt = optimization_scores.get(i, 0.5)
                lex = _lexical_score(text_lower, requirements)
                exact = _exact_match_score(text_lower, query)
                excl = exclusion_scores.get(i, 0.0)

                final = (
                    weights["semantic"] * sem
                    + weights["structured"] * struct
                    + weights["relationship"] * rel
                    + weights["optimization"] * opt
                    + weights["lexical"] * lex
                    + weights["exact"] * exact
                    - EXCLUSION_PENALTY_WEIGHT * excl
                )

                # Structural-coverage penalty: struct/rel already measure
                # how well this chunk satisfies the query's extracted
                # structure (required/important concepts+goals+constraints,
                # relationships), but the additive weights above cap how
                # much they alone can move final_score. A chunk that is
                # only broadly/genuinely on-topic (high sem) but weak on
                # struct/rel would otherwise still be able to outrank a
                # chunk that satisfies the query's complete structure. This
                # multiplier can only shrink final_score toward
                # STRUCTURE_COVERAGE_MIN_MULTIPLIER as coverage -> 0; full
                # coverage (=1.0) leaves final_score unchanged - it never
                # boosts a chunk beyond what the blend above already gave
                # it. No-op (structure_coverage=1.0) when the query has
                # neither a structured sentence nor relationships to check
                # coverage against.
                if has_struct or has_rel:
                    coverage_signals = [
                        v
                        for v, present in ((struct, has_struct), (rel, has_rel))
                        if present
                    ]
                    structure_coverage = sum(coverage_signals) / len(coverage_signals)
                    final *= (
                        STRUCTURE_COVERAGE_MIN_MULTIPLIER
                        + (1 - STRUCTURE_COVERAGE_MIN_MULTIPLIER) * structure_coverage
                    )
                else:
                    structure_coverage = 1.0

                final = max(0.0, final)

                breakdown = ScoreBreakdown(
                    semantic_score=sem,
                    structured_score=struct,
                    relationship_score=rel,
                    optimization_score=opt,
                    lexical_score=lex,
                    exact_match=exact,
                    exclusion_penalty=excl,
                    structure_coverage=structure_coverage,
                    final_score=final,
                    weights_used=weights,
                )
            else:
                final = semantic_scores[i]
                breakdown = ScoreBreakdown(
                    semantic_score=final, final_score=final, weights_used=weights
                )

            blended.append((final, chunk, breakdown))

        # exclusion_penalty is only ever computed for the fine-stage
        # subset above (non-fine-stage chunks default to 0.0, meaning
        # "never evaluated", not "cleared") - see
        # EXCLUSION_HARD_FILTER_THRESHOLD in app/config.py for why this
        # must stay disabled until exclusion evaluation covers every
        # candidate that could be hard-filtered.
        if EXCLUSION_HARD_FILTER_THRESHOLD is not None:
            blended = [
                item
                for item in blended
                if item[2].exclusion_penalty < EXCLUSION_HARD_FILTER_THRESHOLD
            ]

        blended.sort(key=lambda item: item[0], reverse=True)
        return blended

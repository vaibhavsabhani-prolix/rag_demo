"""
Query Understanding Parser

Splits a natural-language patent search query into:
1. semantic_query (pure vector topic)
2. metadata_filters (structured MetadataFilter objects)

The LLM is shown the FIELD_MAPPING allowlist directly (as "CODE: description"
lines, using the human-readable names from metadata_field_codes.py) and asked
to answer using those codes - e.g. a query mentioning "published in 2008"
should come back as {"field": "PY", "operator": "equals", "value": "2008"}.
Application code then just looks the code up in CODE_TO_FIELD and normalizes
the value; it never has to guess which field a piece of text refers to.

Falls back to a pure-semantic query (no metadata filters) if the remote
LLM is unavailable or fails to produce valid JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import (
    QUERY_LLM_REMOTE_API_KEY,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
)
from app.query_understanding.field_mapping import CODE_TO_FIELD, FIELD_MAPPING
from app.query_understanding.models import (
    CandidateFilter,
    Concept,
    Constraint,
    Goal,
    MetadataFilter,
    OptimizationTarget,
    ParsedQuery,
    QuestionIntent,
    RankingWeights,
    Relationship,
    Requirement,
)
from app.query_understanding.normalizer import normalize_country, normalize_org_name
from app.query_understanding.prompt import build_prompt

_COUNTRY_FIELDS = {
    "application_country",
    "publication_country_code",
    "priority_country",
}
_ORG_FIELDS = {
    "current_assignee_normalized",
    "current_assignee_standardized",
    "original_assignee",
    "applicant_first_organization",
    "assignee_applicant_original_with_address",
    "assignee_applicant_standardized_with_address",
}

_ALLOWED_OPERATORS = {
    "equals",
    "contains",
    "not_equals",
    "not_contains",
    "gt",
    "gte",
    "lt",
    "lte",
}

_QUESTION_PREFIXES = (
    "what",
    "which",
    "how",
    "why",
    "where",
    "when",
    "who",
    "whom",
    "whose",
)

_QUESTION_PHRASES = (
    "what properties",
    "which properties",
    "properties determined",
    "problem solved",
    "advantages of",
    "method used",
    "methods used",
    "components used",
    "used to",
    "used for",
    "based on",
)


def resolve_filter(field_code: Any, operator: Any, value: Any) -> MetadataFilter | None:
    """
    Resolve an LLM-emitted (field_code, operator, value) triple to a
    validated MetadataFilter, or None if it can't be trusted.

    field_code must be one of the codes the LLM was shown (CODE_TO_FIELD) -
    anything else is dropped, never invented into a new field.
    """

    field_name = CODE_TO_FIELD.get(str(field_code).strip().upper())
    if field_name is None:
        return None

    spec = FIELD_MAPPING[field_name]

    operator = str(operator).strip().lower() if operator else "equals"
    if operator not in _ALLOWED_OPERATORS or operator not in spec["operators"]:
        operator = spec["operators"][0]

    norm_value = _normalize_value(field_name, spec, value)
    if norm_value is None:
        return None

    return MetadataFilter(field=field_name, operator=operator, value=norm_value)


def _normalize_value(field_name: str, spec: dict, value: Any) -> Any:
    val_str = str(value).strip() if value is not None else ""
    if not val_str:
        return None

    if field_name in _COUNTRY_FIELDS:
        return normalize_country(val_str)

    if field_name in _ORG_FIELDS:
        return normalize_org_name(val_str) or None

    if spec["type"] == "enum":
        for allowed in spec["values"]:
            if val_str.lower() == allowed.lower():
                return allowed
        return None

    if spec["type"] in ("number", "array_number"):
        try:
            num = float(val_str)
        except ValueError:
            return None
        return int(num) if num.is_integer() else num

    return val_str


# ==============================================================
# Dynamic requirements structure (concepts/goals/constraints/
# optimization/exclusions/relationships/requirements/ranking_weights)
#
# Purely additive to the semantic_query/filters extraction above -
# parsed leniently since it feeds Reranker's blended scoring (see
# app/reranker.py), not FIELD_MAPPING-validated metadata truth. A
# malformed entry is dropped rather than raising, and a missing/older
# LLM response (no such keys) simply yields empty lists / defaults.
# ==============================================================


def _as_float(value: Any, default: float = 0.5) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, num))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _parse_concepts(raw: Any) -> list[Concept]:
    concepts: list[Concept] = []
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        concepts.append(
            Concept(
                id=str(item.get("id", "")),
                text=str(item["text"]).strip(),
                role=str(item.get("role", "")).strip(),
                importance=_as_float(item.get("importance")),
                required=_as_bool(item.get("required", False)),
                semantic_variants=_as_str_list(item.get("semantic_variants")),
            )
        )
    return concepts


def _parse_goals(raw: Any) -> list[Goal]:
    goals: list[Goal] = []
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        goals.append(
            Goal(
                id=str(item.get("id", "")),
                text=str(item["text"]).strip(),
                importance=_as_float(item.get("importance")),
                required=_as_bool(item.get("required", False)),
                keywords=_as_str_list(item.get("keywords")),
            )
        )
    return goals


def _parse_constraints(raw: Any) -> list[Constraint]:
    constraints: list[Constraint] = []
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        constraints.append(
            Constraint(
                id=str(item.get("id", "")),
                text=str(item["text"]).strip(),
                type=str(item.get("type", "")).strip(),
                importance=_as_float(item.get("importance")),
                required=_as_bool(item.get("required", False)),
                keywords=_as_str_list(item.get("keywords")),
            )
        )
    return constraints


def _parse_optimization(raw: Any) -> list[OptimizationTarget]:
    targets: list[OptimizationTarget] = []
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("property"):
            continue
        direction = str(item.get("direction", "maximize")).strip().lower()
        if direction not in ("maximize", "minimize"):
            direction = "maximize"
        targets.append(
            OptimizationTarget(
                id=str(item.get("id", "")),
                property=str(item["property"]).strip(),
                direction=direction,
                importance=_as_float(item.get("importance")),
            )
        )
    return targets


def _parse_relationships(raw: Any) -> list[Relationship]:
    relationships: list[Relationship] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target:
            continue
        relationships.append(
            Relationship(
                source=source,
                relation=str(item.get("relation", "")).strip(),
                target=target,
                importance=_as_float(item.get("importance")),
            )
        )
    return relationships


def _parse_requirements(raw: Any) -> list[Requirement]:
    requirements: list[Requirement] = []
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("description"):
            continue
        requirements.append(
            Requirement(
                id=str(item.get("id", "")),
                description=str(item["description"]).strip(),
                type=str(item.get("type", "")).strip(),
                importance=_as_float(item.get("importance")),
                required=_as_bool(item.get("required", False)),
                evaluation_hint=str(item.get("evaluation_hint", "")).strip(),
                keywords=_as_str_list(item.get("keywords")),
            )
        )
    return requirements


def _parse_ranking_weights(raw: Any) -> RankingWeights:
    if not isinstance(raw, dict):
        return RankingWeights()
    return RankingWeights(
        semantic_relevance=_as_float(raw.get("semantic_relevance"), 1.0),
        requirement_satisfaction=_as_float(raw.get("requirement_satisfaction"), 0.0),
        relationship_satisfaction=_as_float(raw.get("relationship_satisfaction"), 0.0),
        constraint_satisfaction=_as_float(raw.get("constraint_satisfaction"), 0.0),
        evidence_strength=_as_float(raw.get("evidence_strength"), 0.0),
        exact_match=_as_float(raw.get("exact_match"), 0.0),
    )


def _fallback_is_question(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if "?" in q:
        return True
    if any(q.startswith(prefix + " ") for prefix in _QUESTION_PREFIXES):
        return True
    return any(phrase in q for phrase in _QUESTION_PHRASES)


def _extract_question_fields(
    query: str, llm_res: dict[str, Any]
) -> tuple[bool, QuestionIntent | None]:
    raw_is_question = llm_res.get("is_question")
    intent_obj = llm_res.get("question_intent")

    question_intent: QuestionIntent | None = None
    if isinstance(intent_obj, dict):
        target = str(intent_obj.get("target") or "").strip()
        expected_answer_type = str(intent_obj.get("expected_answer_type") or "").strip()
        answer_criteria = str(intent_obj.get("answer_criteria") or "").strip()

        if target or expected_answer_type or answer_criteria:
            question_intent = QuestionIntent(
                is_question=True,
                target=target,
                expected_answer_type=expected_answer_type,
                answer_criteria=answer_criteria,
            )

    if isinstance(raw_is_question, bool):
        is_question = raw_is_question
    elif isinstance(raw_is_question, str):
        is_question = raw_is_question.strip().lower() in ("true", "yes", "1")
    else:
        is_question = question_intent is not None or _fallback_is_question(query)

    if not is_question:
        return False, None

    if question_intent is None:
        question_intent = QuestionIntent(
            is_question=True,
            target="",
            expected_answer_type="",
            answer_criteria="",
        )

    return True, question_intent


class QueryUnderstanding:
    """
    LLM-based Query Understanding: the LLM is given the field-code
    allowlist and answers directly in those terms. Falls back to a
    pure-semantic query (no filters) if the remote LLM is unavailable
    or its output can't be parsed as valid JSON.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

        self._remote_client = None
        self._remote_available = False

        if self.use_llm:
            self._check_remote_llm()

    def _check_remote_llm(self):
        """Check whether the remote Qwen server is available."""
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=QUERY_LLM_REMOTE_BASE_URL,
                api_key=QUERY_LLM_REMOTE_API_KEY,
                timeout=60.0,
            )

            # Lightweight connectivity/model check.
            client.models.list()

            self._remote_client = client
            self._remote_available = True

            print(f"Remote Query LLM available: {QUERY_LLM_REMOTE_MODEL}")

        except Exception as e:
            self._remote_client = None
            self._remote_available = False

            print(f"[QueryUnderstanding] Remote Qwen unavailable: {e}")

    def _call_remote_llm(self, query: str) -> dict | None:
        """
        Send the query-understanding prompt to the remote Qwen model
        through its OpenAI-compatible API.
        """

        if not self._remote_client:
            return None

        prompt = build_prompt(query)

        try:
            response = self._remote_client.chat.completions.create(
                model=QUERY_LLM_REMOTE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                # temperature=0,
                # max_tokens=512,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            content = response.choices[0].message.content

            if not content:
                self._last_remote_error = "Remote model returned empty content"
                return None

            # Remove Qwen thinking blocks if present
            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()

            # Remove Markdown code fences
            cleaned = re.sub(
                r"```(?:json)?\s*",
                "",
                cleaned,
            ).strip()

            # Extract JSON object
            match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    self._last_remote_error = (
                        f"Invalid JSON in response: {e}. Raw content: {content[:500]}"
                    )
                    return None

            self._last_remote_error = (
                f"No JSON object found in remote response. Raw content: {content[:500]}"
            )
            return None

        except Exception as e:
            self._last_remote_error = str(e)
            return None

    def _call_llm(self, query: str) -> dict | None:
        """
        Query Understanding LLM call: uses the remote Qwen server,
        retried once on failure.
        """

        if not self.use_llm:
            return None

        if not self._remote_available:
            print("[QueryUnderstanding] Remote LLM is not available.")
            return None

        print(f"Using remote LLM: {QUERY_LLM_REMOTE_MODEL}")

        # First remote attempt
        self._last_remote_error = None
        result = self._call_remote_llm(query)

        if result is not None:
            return result

        first_error = self._last_remote_error or "unknown error"

        print(f"[QueryUnderstanding] Remote Qwen first attempt failed: {first_error}")

        # ------------------------------------------------------
        # Retry remote once
        # ------------------------------------------------------

        print("[QueryUnderstanding] Retrying remote Qwen...")

        self._last_remote_error = None
        result = self._call_remote_llm(query)

        if result is not None:
            print("[QueryUnderstanding] Remote Qwen retry succeeded.")
            return result

        second_error = self._last_remote_error or "unknown error"

        print(f"[QueryUnderstanding] Remote Qwen retry failed: {second_error}.")

        # Remote is considered unavailable only after retry fails
        self._remote_available = False

        return None

    def parse(self, query: str) -> ParsedQuery:
        original = query
        query = query.strip()

        if not query:
            return ParsedQuery(
                original_query=original, semantic_query=original, metadata_filters=[]
            )

        llm_res = self._call_llm(query)

        if (
            not llm_res
            or not isinstance(llm_res, dict)
            or "semantic_query" not in llm_res
        ):
            is_question = _fallback_is_question(query)
            return ParsedQuery(
                original_query=original,
                semantic_query=query,
                metadata_filters=[],
                is_question=is_question,
                question_intent=(
                    QuestionIntent(is_question=True) if is_question else None
                ),
            )

        raw_semantic = str(llm_res.get("semantic_query") or "").strip()

        candidate_filters: list[CandidateFilter] = []
        metadata_filters: list[MetadataFilter] = []

        for item in llm_res.get("filters", []) or []:
            if not isinstance(item, dict) or "value" not in item:
                continue

            # Defends against an observed remote-LLM glitch: on a
            # 2nd-or-later filter in the array, the "field" or
            # "operator" key occasionally comes back corrupted (e.g.
            # "", ".field", "=") while its intended value (the real
            # field code / operator) survives under that stray key.
            # Recover missing expected keys from whatever stray keys
            # are left, in order - resolve_filter still validates the
            # recovered field code against CODE_TO_FIELD, so a
            # genuinely malformed item is dropped either way.
            stray_values = [
                v for k, v in item.items() if k not in ("field", "operator", "value")
            ]

            field_code = item.get("field")
            if field_code is None and stray_values:
                field_code = stray_values.pop(0)

            if field_code is None:
                continue

            operator = item.get("operator")
            if operator is None and stray_values:
                operator = stray_values.pop(0)
            operator = operator or "equals"

            value = item["value"]

            candidate_filters.append(
                CandidateFilter(meaning=str(field_code), value=value)
            )

            resolved = resolve_filter(field_code, operator, value)
            if resolved is not None:
                metadata_filters.append(resolved)

        # An empty semantic_query is the LLM's deliberate "this query is
        # pure metadata filters, no real topic" signal (see prompt Rule
        # 9B) - trust it only if it actually produced filters, otherwise
        # an empty semantic_query would mean "search for nothing" and we
        # fall back to the raw query text instead.
        semantic_query = raw_semantic if (raw_semantic or metadata_filters) else query

        is_question, question_intent = _extract_question_fields(query, llm_res)

        return ParsedQuery(
            original_query=original,
            semantic_query=semantic_query,
            candidate_filters=candidate_filters,
            metadata_filters=metadata_filters,
            intent=str(llm_res.get("intent") or "").strip(),
            query_type=_as_str_list(llm_res.get("query_type")),
            concepts=_parse_concepts(llm_res.get("concepts")),
            goals=_parse_goals(llm_res.get("goals")),
            constraints=_parse_constraints(llm_res.get("constraints")),
            optimization=_parse_optimization(llm_res.get("optimization")),
            exclusions=_as_str_list(llm_res.get("exclusions")),
            relationships=_parse_relationships(llm_res.get("relationships")),
            requirements=_parse_requirements(llm_res.get("requirements")),
            ranking_weights=_parse_ranking_weights(llm_res.get("ranking_weights")),
            is_question=is_question,
            question_intent=question_intent,
        )

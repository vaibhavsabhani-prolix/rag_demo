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

Falls back to a pure-semantic query (no metadata filters) if the local LLM
is unavailable or fails to produce valid JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import QUERY_LLM_MODEL
from app.query_understanding.field_mapping import CODE_TO_FIELD, FIELD_MAPPING
from app.query_understanding.metadata_field_codes import METADATA_FIELD_CODES
from app.query_understanding.models import CandidateFilter, MetadataFilter, ParsedQuery
from app.query_understanding.normalizer import normalize_country, normalize_org_name

_COUNTRY_FIELDS = {"application_country", "publication_country_code", "priority_country"}
_ORG_FIELDS = {
    "current_assignee_normalized",
    "current_assignee_standardized",
    "original_assignee",
    "applicant_first_organization",
    "assignee_applicant_original_with_address",
    "assignee_applicant_standardized_with_address",
}

_FIELD_LIST_PROMPT = "\n".join(
    f"{code}: {METADATA_FIELD_CODES.get(code, code)} ({FIELD_MAPPING[name]['type']})"
    for code, name in sorted(CODE_TO_FIELD.items())
)

_ALLOWED_OPERATORS = {"equals", "contains", "gt", "gte", "lt", "lte"}


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


class QueryUnderstanding:
    """
    LLM-based Query Understanding: the LLM is given the field-code
    allowlist and answers directly in those terms. Falls back to a
    pure-semantic query (no filters) if the local LLM is unavailable
    or its output can't be parsed as valid JSON.
    """

    def __init__(self, model_name: str | None = None, use_llm: bool = True):
        self.model_name = model_name or QUERY_LLM_MODEL
        self.use_llm = use_llm
        self._llm_model = None
        self._llm_tokenizer = None
        self._llm_initialized = False

    def _init_llm(self):
        if self._llm_initialized:
            return
        self._llm_initialized = True
        if not self.use_llm:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading local LLM for Query Understanding: {self.model_name}")
            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            print("Local LLM loaded successfully.\n")
        except Exception as e:
            print(f"[QueryUnderstanding] Warning: Local LLM ({self.model_name}) not loaded: {e}. Falling back to pure-semantic parsing.")
            self._llm_model = None
            self._llm_tokenizer = None

    def _build_prompt(self, query: str) -> str:
        return f"""<|im_start|>system
You are the query-understanding component of a patent semantic search engine.

Split the user's query into:

1. semantic_query
   The actual invention/topic to search for.

2. filters
   Metadata constraints, each as {{"field", "operator", "value"}}.

Field codes (the ONLY values "field" may take):
{_FIELD_LIST_PROMPT}

Rules:
- "field" must be exactly one of the codes above. Never invent a code.
- "operator" is one of: equals, contains, gt, gte, lt, lte.
- Each code's type is shown in parentheses. A bare year or number (e.g.
  "2008", "after 2018") MUST go to a field of type "number" or
  "array_number" - never to a "string"/"array_string" classification
  code (CPC/IPC/etc.) or a country code, even if the sentence also
  mentions a place, status, or classification elsewhere.
- "value" must be copied verbatim from the query - never invent a value
  that isn't in the query text.
- Remove recognized filters from semantic_query.
- If nothing matches a filter, return an empty filters list.
- Return valid JSON only. No Markdown. No explanation.

Example:
User: bottle designs patented by Coca Cola in the US
Output: {{"semantic_query": "bottle designs", "filters": [{{"field": "CAN_EN", "operator": "contains", "value": "Coca Cola"}}, {{"field": "AC", "operator": "equals", "value": "US"}}]}}

Example:
User: bottle design patents published in Japan after 2018
Output: {{"semantic_query": "bottle design", "filters": [{{"field": "PNC", "operator": "equals", "value": "Japan"}}, {{"field": "PY", "operator": "gt", "value": "2018"}}]}}

Example:
User: what problems with existing antimalarial compounds are discussed in patents published in 2008?
Output: {{"semantic_query": "problems with existing antimalarial compounds", "filters": [{{"field": "PY", "operator": "equals", "value": "2008"}}]}}

Example:
User: bottle design
Output: {{"semantic_query": "bottle design", "filters": []}}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

    def _call_llm(self, query: str) -> dict | None:
        self._init_llm()
        if self._llm_model is None or self._llm_tokenizer is None:
            return None

        prompt = self._build_prompt(query)

        try:
            inputs = self._llm_tokenizer(prompt, return_tensors="pt")
            outputs = self._llm_model.generate(
                **inputs, max_new_tokens=300, temperature=0.01, do_sample=False
            )
            response = self._llm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            # Strip Markdown code fences (```json ... ```) if the LLM wraps output
            cleaned = re.sub(r"```(?:json)?\s*", "", response)
            cleaned = cleaned.strip()

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"[QueryUnderstanding] LLM execution/parsing failed: {e}")

        return None

    def parse(self, query: str) -> ParsedQuery:
        original = query
        query = query.strip()

        if not query:
            return ParsedQuery(original_query=original, semantic_query=original, metadata_filters=[])

        llm_res = self._call_llm(query)

        if not llm_res or not isinstance(llm_res, dict) or "semantic_query" not in llm_res:
            return ParsedQuery(original_query=original, semantic_query=query, metadata_filters=[])

        semantic_query = str(llm_res.get("semantic_query") or query).strip() or query

        candidate_filters: list[CandidateFilter] = []
        metadata_filters: list[MetadataFilter] = []

        for item in llm_res.get("filters", []) or []:
            if not isinstance(item, dict) or "field" not in item or "value" not in item:
                continue

            field_code = item["field"]
            operator = item.get("operator", "equals")
            value = item["value"]

            candidate_filters.append(CandidateFilter(meaning=str(field_code), value=value))

            resolved = resolve_filter(field_code, operator, value)
            if resolved is not None:
                metadata_filters.append(resolved)

        return ParsedQuery(
            original_query=original,
            semantic_query=semantic_query,
            candidate_filters=candidate_filters,
            metadata_filters=metadata_filters,
        )

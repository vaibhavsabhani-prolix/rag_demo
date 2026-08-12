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

from app.config import (
    QUERY_LLM_MODEL,
    QUERY_LLM_REMOTE_API_KEY,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    USE_REMOTE_LLM,
)
from app.query_understanding.field_mapping import CODE_TO_FIELD, FIELD_MAPPING
from app.query_understanding.models import CandidateFilter, MetadataFilter, ParsedQuery
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

        # Local model
        self._llm_model = None
        self._llm_tokenizer = None
        self._llm_initialized = False

        # Remote model
        self._remote_client = None
        self._remote_available = False

        if self.use_llm and USE_REMOTE_LLM:
            self._check_remote_llm()

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
            print(
                f"[QueryUnderstanding] Warning: Local LLM ({self.model_name}) not loaded: {e}. Falling back to pure-semantic parsing."
            )
            self._llm_model = None
            self._llm_tokenizer = None

    def _check_remote_llm(self):
        """
        Check whether the remote Qwen server is available.

        If available, remote Qwen becomes the primary model.
        If unavailable, the local Qwen model will be used as fallback.
        """
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

            print(f"Remote Query LLM available: " f"{QUERY_LLM_REMOTE_MODEL}")

        except Exception as e:
            self._remote_client = None
            self._remote_available = False

            print(
                "[QueryUnderstanding] Remote Qwen unavailable. "
                "Local model will be used as fallback."
            )

    def _call_local_llm(self, query: str) -> dict | None:
        self._init_llm()
        if self._llm_model is None or self._llm_tokenizer is None:
            return None

        prompt = build_prompt(query)

        try:
            inputs = self._llm_tokenizer(prompt, return_tensors="pt")
            outputs = self._llm_model.generate(
                **inputs, max_new_tokens=300, temperature=0.01, do_sample=False
            )
            response = self._llm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
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
                temperature=0,
                max_tokens=512,
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
                        f"Invalid JSON in response: {e}. "
                        f"Raw content: {content[:500]}"
                    )
                    return None

            self._last_remote_error = (
                "No JSON object found in remote response. "
                f"Raw content: {content[:500]}"
            )
            return None

        except Exception as e:
            self._last_remote_error = str(e)
            return None

    def _call_llm(self, query: str) -> dict | None:
        """
        Query Understanding LLM routing, controlled by USE_REMOTE_LLM:

        - True: use the remote Qwen server (retried once on failure).
        - False: use the local Qwen model.
        """

        if not self.use_llm:
            return None

        if not USE_REMOTE_LLM:
            print(f"Using local LLM: {self.model_name}")
            return self._call_local_llm(query)

        # ----------------------------------------------------------
        # Remote Qwen
        # ----------------------------------------------------------

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

        print(
            f"[QueryUnderstanding] Remote Qwen first attempt failed: "
            f"{first_error}"
        )

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

        print(
            f"[QueryUnderstanding] Remote Qwen retry failed: "
            f"{second_error}."
        )

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
            return ParsedQuery(
                original_query=original, semantic_query=query, metadata_filters=[]
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

        return ParsedQuery(
            original_query=original,
            semantic_query=semantic_query,
            candidate_filters=candidate_filters,
            metadata_filters=metadata_filters,
        )

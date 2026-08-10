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
)

from app.config import QUERY_LLM_MODEL
from app.query_understanding.field_mapping import CODE_TO_FIELD, FIELD_MAPPING
from app.query_understanding.metadata_field_codes import METADATA_FIELD_CODES
from app.query_understanding.models import CandidateFilter, MetadataFilter, ParsedQuery
from app.query_understanding.normalizer import normalize_country, normalize_org_name

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

        # Local model
        self._llm_model = None
        self._llm_tokenizer = None
        self._llm_initialized = False

        # Remote model
        self._remote_client = None
        self._remote_available = False

        # Try remote first.
        if self.use_llm:
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

    def _build_prompt(self, query: str) -> str:
        return f"""<|im_start|>system
You are the Query Understanding component of a patent semantic search engine.

Your job is to convert the user's natural-language patent search query into
STRICT JSON containing:

1. semantic_query
   The actual invention, technology, subject, or concept that should be
   searched using semantic/vector search.

2. filters
   Metadata constraints found in the user's query.

IMPORTANT:
You MUST select filter fields ONLY from the field-code allowlist below.
NEVER invent a field code, field name, or metadata field.

============================================================
ALLOWED METADATA FIELD CODES
============================================================

{_FIELD_LIST_PROMPT}

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON:

{{
  "semantic_query": "...",
  "filters": [
    {{
      "field": "FIELD_CODE",
      "operator": "equals|contains|gt|gte|lt|lte",
      "value": "VALUE"
    }}
  ]
}}

If there are no metadata filters:

{{
  "semantic_query": "...",
  "filters": []
}}

Do NOT return Markdown.
Do NOT return explanations.
Do NOT return comments.
Do NOT return additional keys.

============================================================
CORE RULES
============================================================

RULE 1 — FIELD ALLOWLIST

The "field" value MUST be exactly one of the field codes listed above.

Never create a field such as:

- "creator"
- "company"
- "country"
- "year"
- "status"
- "inventor_name"

Instead, map the user's language to one of the provided official
field codes.

For example:

"creator" → IN_EN
"inventor" → IN_EN
"priority country" → PRC
"publication year" → PY
"legal status" → LST
"legal state" → ALD

The field code must always come from the provided allowlist.

============================================================
RULE 2 — NATURAL LANGUAGE FIELD MAPPING
============================================================

Use the following semantic mappings when interpreting the user's language.

INVENTOR:

"creator"
"creator of the patent"
"created by"
"inventor"
"invented by"
"made by the inventor"
"patent created by"

→ IN_EN (Inventor English)

Do NOT map "creator" to an assignee or applicant field.

ASSIGNEE:

"assigned to"
"owned by"
"currently owned by"
"current assignee"
"patent owned by"

→ the appropriate Current Assignee field from the allowlist.

APPLICANT:

"applicant"
"filed by"
"application filed by"

→ the appropriate Applicant field from the allowlist.

PRIORITY COUNTRY:

"priority country"
"priority in"
"priority filed in"
"priority from"

→ PRC (Priority Country)

APPLICATION COUNTRY:

"application country"
"country of application"
"filed in"
"application filed in"

→ AC (Application Country)

PUBLICATION COUNTRY:

"publication country"
"published in [country]"
"country published in"

→ PNC (Publication Country Code)

PUBLICATION YEAR:

"publication year"
"published in [year]"
"published during [year]"
"publication date/year"

→ PY (Publication Year)

APPLICATION YEAR:

"application year"
"filing year"
"filed in [year]"

→ AY (Application Year)

PRIORITY YEAR:

"priority year"
"priority in [year]"

→ PRY (Priority Year)

EARLIEST PRIORITY YEAR:

"earliest priority year"

→ EPRY (Earliest Priority Year)

LEGAL STATUS:

"legal status"

→ LST (Legal Status)

Allowed values:
- Filed
- Granted
- Ceased

LEGAL STATE:

"legal state"

→ ALD (Legal State)

Allowed values:
- Alive
- Dead

IMPORTANT:
Legal Status and Legal State are DIFFERENT fields.

"Filed" is a Legal Status value.

"Granted" is a Legal Status value.

"Ceased" is a Legal Status value.

"Alive" is a Legal State value.

"Dead" is a Legal State value.

NEVER swap LST and ALD.

CPC:

"cpc"
"cpc classification"
"cpc class"

→ CPC-related field from the allowlist according to the wording.

IPC:

"ipc"
"ipc classification"
"ipc class"

→ IPC-related field from the allowlist according to the wording.

============================================================
RULE 3 — OPERATOR SELECTION
============================================================

Use operators according to the meaning of the user's query.

equals:
Use when the user specifies an exact value.

Examples:
"published in 2008"
"legal status is Filed"
"legal state is Alive"

contains:
Use for textual membership/substring fields such as inventor,
assignee, applicant, CPC, IPC, and country arrays where supported.

Examples:
"invented by RUSCH CHRISTOPH"
"owned by Coca Cola"
"priority country China"

gt:
Use for "after", "greater than", "later than".

Example:
"published after 2018"

→ PY gt 2018

gte:
Use for "from", "since", "at least", "starting from", when the
starting boundary is inclusive.

Example:
"published from 2005"

→ PY gte 2005

lt:
Use for "before", "less than", "earlier than".

Example:
"published before 2010"

→ PY lt 2010

lte:
Use for "up to", "until", "no later than", when the ending boundary
is inclusive.

Example:
"published up to 2010"

→ PY lte 2010

============================================================
RULE 4 — YEAR RANGES
============================================================

This is VERY IMPORTANT.

When the user specifies a range such as:

"from 2005 to 2010"
"between 2005 and 2010"
"2005 to 2010"
"published between 2005 and 2010"
"published from 2005 through 2010"

create TWO filters.

Example:

User:
"patents published from 2005 to 2010"

Correct:

{{
  "semantic_query": "patents",
  "filters": [
    {{
      "field": "PY",
      "operator": "gte",
      "value": "2005"
    }},
    {{
      "field": "PY",
      "operator": "lte",
      "value": "2010"
    }}
  ]
}}

Do NOT produce:

{{
  "field": "PY",
  "operator": "equals",
  "value": "2005"
}}

and:

{{
  "field": "PY",
  "operator": "equals",
  "value": "2010"
}}

A range means LOWER BOUND + UPPER BOUND.

============================================================
RULE 5 — MULTIPLE FILTERS
============================================================

A user's query can contain many filters.

Identify ALL filters that are clearly expressed in the query.

Do not stop after finding the first filter.

Example:

User:
"water patents published from 2005 to 2010 invented by RUSCH CHRISTOPH
with priority country China and legal status Filed and legal state Alive"

This contains SIX filter conditions:

1. Publication Year >= 2005
2. Publication Year <= 2010
3. Inventor contains RUSCH CHRISTOPH
4. Priority Country contains China
5. Legal Status equals Filed
6. Legal State equals Alive

Correct output:

{{
  "semantic_query": "water patents",
  "filters": [
    {{
      "field": "PY",
      "operator": "gte",
      "value": "2005"
    }},
    {{
      "field": "PY",
      "operator": "lte",
      "value": "2010"
    }},
    {{
      "field": "IN_EN",
      "operator": "contains",
      "value": "RUSCH CHRISTOPH"
    }},
    {{
      "field": "PRC",
      "operator": "contains",
      "value": "China"
    }},
    {{
      "field": "LST",
      "operator": "equals",
      "value": "Filed"
    }},
    {{
      "field": "ALD",
      "operator": "equals",
      "value": "Alive"
    }}
  ]
}}

============================================================
RULE 6 — DO NOT CONFUSE INVENTOR, ASSIGNEE, AND APPLICANT
============================================================

These are different concepts.

"creator"
"inventor"
"invented by"

→ IN_EN

"assigned to"
"owned by"
"current owner"

→ Current Assignee field

"applicant"
"filed by"

→ Applicant field

Example:

"patent created by John Smith"

MUST NOT become an assignee filter.

It must become:

{{
  "field": "IN_EN",
  "operator": "contains",
  "value": "John Smith"
}}

============================================================
RULE 7 — LEGAL STATUS VS LEGAL STATE
============================================================

Never confuse these two.

Example:

"legal status is Filed"

→

{{
  "field": "LST",
  "operator": "equals",
  "value": "Filed"
}}

Example:

"legal state is Alive"

→

{{
  "field": "ALD",
  "operator": "equals",
  "value": "Alive"
}}

Example:

"Filed and Alive"

when the user explicitly refers to legal status/state:

→

{{
  "field": "LST",
  "operator": "equals",
  "value": "Filed"
}},
{{
  "field": "ALD",
  "operator": "equals",
  "value": "Alive"
}}

============================================================
RULE 8 — VALUES
============================================================

The "value" must come from the user's query.

Do NOT invent a person's name, company name, year, country, status,
classification, or other value.

For example:

User:
"patents invented by RUSCH CHRISTOPH"

Use:

"value": "RUSCH CHRISTOPH"

Do not change the person to another name.

Country names may be normalized later by application code.
Therefore return the country value as it appears in the user's query.

Example:

"China" → value "China"

"United States" → value "United States"

Do not convert countries to ISO codes yourself.

============================================================
RULE 9 — SEMANTIC QUERY
============================================================

semantic_query must contain only the actual invention/topic/concept
that should be sent to the embedding model.

Remove recognized metadata filter phrases from semantic_query.

Example:

User:
"bottle designs patented by Coca Cola in the US"

Correct:

"semantic_query": "bottle designs"

NOT:

"bottle designs patented by Coca Cola in the US"

Example:

User:
"water related patents invented by RUSCH CHRISTOPH"

Correct:

"semantic_query": "water related patents"

Example:

User:
"patents about pressure control in bottles published in Japan after 2018"

Correct:

"semantic_query": "pressure control in bottles"

============================================================
RULE 10 — DO NOT INVENT FILTERS
============================================================

If a phrase does not clearly correspond to one of the allowed metadata
fields, do NOT create a filter.

Example:

"red bottle patents"

If there is no allowed metadata field for bottle color:

"filters": []

The phrase "red" should remain part of semantic_query if it describes
the invention/topic.

Example:

"bottle patents with red color"

If color is not an allowed metadata field, do not invent:

"bottle_color"

============================================================
RULE 11 — NUMBERS AND YEARS
============================================================

A year must map to an appropriate year field.

Examples:

"published in 2008"
→ PY equals 2008

"published after 2018"
→ PY gt 2018

"published from 2005"
→ PY gte 2005

"published before 2010"
→ PY lt 2010

"published through 2010"
→ PY lte 2010

"published from 2005 to 2010"
→ PY gte 2005
→ PY lte 2010

Never classify a year as CPC, IPC, country, inventor, assignee,
or another unrelated field.

============================================================
RULE 12 — FILTER FIELD MUST BE AN OFFICIAL CODE
============================================================

Before producing the JSON, internally verify:

1. Every filter has a "field".
2. Every field is one of the supplied field codes.
3. Every operator is one of:
   equals, contains, gt, gte, lt, lte.
4. The operator makes sense for the selected field.
5. Every value came from the user's query.
6. All clearly expressed filters have been extracted.
7. No unsupported filter has been invented.
8. Legal Status and Legal State are not swapped.
9. Inventor/creator is not confused with assignee.
10. Year ranges produce two boundary filters.

============================================================
EXAMPLES
============================================================

Example 1:

User:
"bottle design"

Output:

{{
  "semantic_query": "bottle design",
  "filters": []
}}

Example 2:

User:
"bottle designs patented by Coca Cola"

Output:

{{
  "semantic_query": "bottle designs",
  "filters": [
    {{
      "field": "CAN_EN",
      "operator": "contains",
      "value": "Coca Cola"
    }}
  ]
}}

Example 3:

User:
"bottle designs invented by RUSCH CHRISTOPH"

Output:

{{
  "semantic_query": "bottle designs",
  "filters": [
    {{
      "field": "IN_EN",
      "operator": "contains",
      "value": "RUSCH CHRISTOPH"
    }}
  ]
}}

Example 4:

User:
"bottle patents in the US"

Output:

{{
  "semantic_query": "bottle patents",
  "filters": [
    {{
      "field": "AC",
      "operator": "equals",
      "value": "US"
    }}
  ]
}}

Example 5:

User:
"bottle patents published in Japan after 2018"

Output:

{{
  "semantic_query": "bottle patents",
  "filters": [
    {{
      "field": "PNC",
      "operator": "equals",
      "value": "Japan"
    }},
    {{
      "field": "PY",
      "operator": "gt",
      "value": "2018"
    }}
  ]
}}

Example 6:

User:
"water patents published from 2005 to 2010"

Output:

{{
  "semantic_query": "water patents",
  "filters": [
    {{
      "field": "PY",
      "operator": "gte",
      "value": "2005"
    }},
    {{
      "field": "PY",
      "operator": "lte",
      "value": "2010"
    }}
  ]
}}

Example 7:

User:
"water patents with priority country China"

Output:

{{
  "semantic_query": "water patents",
  "filters": [
    {{
      "field": "PRC",
      "operator": "contains",
      "value": "China"
    }}
  ]
}}

Example 8:

User:
"patents with legal status Filed and legal state Alive"

Output:

{{
  "semantic_query": "patents",
  "filters": [
    {{
      "field": "LST",
      "operator": "equals",
      "value": "Filed"
    }},
    {{
      "field": "ALD",
      "operator": "equals",
      "value": "Alive"
    }}
  ]
}}

Example 9:

User:
"water related patents published from 2005 to 2010 invented by RUSCH CHRISTOPH with priority country China and legal status Filed and legal state Alive"

Output:

{{
  "semantic_query": "water related patents",
  "filters": [
    {{
      "field": "PY",
      "operator": "gte",
      "value": "2005"
    }},
    {{
      "field": "PY",
      "operator": "lte",
      "value": "2010"
    }},
    {{
      "field": "IN_EN",
      "operator": "contains",
      "value": "RUSCH CHRISTOPH"
    }},
    {{
      "field": "PRC",
      "operator": "contains",
      "value": "China"
    }},
    {{
      "field": "LST",
      "operator": "equals",
      "value": "Filed"
    }},
    {{
      "field": "ALD",
      "operator": "equals",
      "value": "Alive"
    }}
  ]
}}

============================================================
FINAL INSTRUCTION
============================================================

Now analyze the user's query.

Return ONLY the JSON object.

No Markdown.
No explanation.
No reasoning.
No extra text.

<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""

    def _call_local_llm(self, query: str) -> dict | None:
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

        prompt = self._build_prompt(query)

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
                max_tokens=2048,
            )

            content = response.choices[0].message.content

            if not content:
                self._last_remote_error = "Remote model returned empty content"
                return None

            # Strip <think>...</think> blocks (Qwen 3 reasoning mode)
            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()

            # Strip Markdown code fences
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

            match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as je:
                    self._last_remote_error = (
                        f"Invalid JSON in response: {je}. Raw content: {content[:500]}"
                    )
                    return None

            self._last_remote_error = (
                f"No JSON object found in response. Raw content: {content[:500]}"
            )

        except Exception as e:
            self._last_remote_error = str(e)

        return None

    def _call_llm(self, query: str) -> dict | None:
        """
        Query Understanding LLM routing:

        1. Try remote Qwen first.
        2. If remote fails, fall back to local Qwen.
        3. If local also fails, return None.
        """

        if not self.use_llm:
            return None

        # ----------------------------------------------------------
        # 1. Remote Qwen
        # ----------------------------------------------------------

        if self._remote_available:
            print(f"Using remote LLM: {QUERY_LLM_REMOTE_MODEL}")
            self._last_remote_error = None
            result = self._call_remote_llm(query)

            if result is not None:
                return result

            # Remote was available before, but failed during
            # this request. Disable it and fall back to local.
            self._remote_available = False

            error_detail = self._last_remote_error or "unknown error"
            print(
                f"[QueryUnderstanding] Remote Qwen request failed: {error_detail}. "
                "Falling back to local Qwen."
            )

        # ----------------------------------------------------------
        # 2. Local Qwen fallback
        # ----------------------------------------------------------

        print(f"Using local LLM: {self.model_name}")
        return self._call_local_llm(query)

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

        semantic_query = str(llm_res.get("semantic_query") or query).strip() or query

        candidate_filters: list[CandidateFilter] = []
        metadata_filters: list[MetadataFilter] = []

        for item in llm_res.get("filters", []) or []:
            if not isinstance(item, dict) or "field" not in item or "value" not in item:
                continue

            field_code = item["field"]
            operator = item.get("operator", "equals")
            value = item["value"]

            candidate_filters.append(
                CandidateFilter(meaning=str(field_code), value=value)
            )

            resolved = resolve_filter(field_code, operator, value)
            if resolved is not None:
                metadata_filters.append(resolved)

        return ParsedQuery(
            original_query=original,
            semantic_query=semantic_query,
            candidate_filters=candidate_filters,
            metadata_filters=metadata_filters,
        )

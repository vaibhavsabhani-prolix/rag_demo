"""
Query Understanding Parser

Splits a natural-language patent search query into:
1. semantic_query (pure vector topic)
2. candidate_filters (extracted natural-language meaning + raw value)
3. metadata_filters (validated MetadataFilter objects mapped strictly against FIELD_MAPPING)

Uses a local Qwen instruction model (configured via QUERY_LLM_MODEL in app/config.py)
for query understanding. The LLM produces candidate meanings + values, which are then
mapped to approved fields in FIELD_MAPPING by application code.

Falls back gracefully to rule-based parsing if the local LLM is unavailable or fails JSON output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import QUERY_LLM_MODEL
from app.query_understanding.field_mapping import FIELD_MAPPING
from app.query_understanding.models import (
    CandidateFilter,
    MetadataFilter,
    ParsedQuery,
)
from app.query_understanding.normalizer import (
    find_country_in_text,
    normalize_country,
    normalize_org_name,
)

# Fields whose value is a country code
_COUNTRY_FIELDS = {"application_country", "publication_country_code", "priority_country"}

_COUNTRY_TO_YEAR_FIELD = {
    "application_country": "application_year",
    "publication_country_code": "publication_year",
    "priority_country": "earliest_priority_year",
}

_COMPARATOR_OPERATORS = {
    "after": "gt",
    "before": "lt",
    "since": "gte",
    "from": "gte",
    "until": "lte",
    "in": "equals",
    "during": "equals",
}
_COMPARATOR_PHRASES = sorted(_COMPARATOR_OPERATORS, key=len, reverse=True)
_COMPARATOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _COMPARATOR_PHRASES) + r")\b", re.IGNORECASE
)

_NUMBER_VALUE_RE = re.compile(
    r"^\s*(?:(" + "|".join(re.escape(w) for w in _COMPARATOR_PHRASES) + r")\s+)?"
    r"(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_ENUM_SYNONYMS = {
    "legal_state": {"Alive": ["active", "alive"], "Dead": ["expired", "dead"]},
    "legal_status": {"Filed": ["pending", "filed"], "Granted": ["granted"], "Ceased": ["abandoned", "ceased"]},
}

_TEXT_CAPTURE_STOPWORDS = {"and", "with", "having", "where", "filed", "published", "from", "in"}
_LEADING_FILLER = {"about", "regarding", "describing", "related", "to", "for", "find", "search", "show", "list", "on"}
_PATENT_WORD_RE = re.compile(r"^(patents?|applications?)\b", re.IGNORECASE)
_TRAILING_PATENT_WORD_RE = re.compile(r"\b(patents?|applications?)$", re.IGNORECASE)
_TRAILING_COUNTRY_RE = re.compile(
    r"\b(?:patents?\s+)?in\s+(?:the\s+)?([A-Za-z][A-Za-z\s]*)$", re.IGNORECASE
)


def _is_year_or_number(val: str) -> bool:
    cleaned = str(val).strip()
    if re.match(r"^\d{4}$", cleaned):
        return True
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def resolve_candidate_filter(meaning: str, value: Any) -> MetadataFilter | None:
    """
    Resolve a candidate filter (meaning, value) to an official MetadataFilter.

    Authoritative FIELD_MAPPING Allowlist Rules:
    1. Field MUST exist in FIELD_MAPPING.
    2. Operator MUST be in FIELD_MAPPING[field]["operators"].
    3. Value MUST be normalized successfully.
    4. If unmapped or uncertain, returns None (never invents a field).
    """
    meaning_lower = meaning.lower().strip()
    val_str = str(value).strip() if value is not None else ""
    if not val_str:
        return None

    target_field = None

    # Step 1: Check explicit alias list in FIELD_MAPPING
    for field_key, spec in FIELD_MAPPING.items():
        aliases = spec.get("aliases", [])
        for alias in aliases:
            if alias.lower() in meaning_lower or meaning_lower == alias.lower():
                # Guard: if this alias maps to a country field but the value
                # looks like a year/number, skip — the year-field variant
                # should match instead (e.g. "filed in 2020" should not
                # resolve to application_country).
                if field_key in _COUNTRY_FIELDS and _is_year_or_number(val_str):
                    continue
                target_field = field_key
                break
        if target_field:
            break

    # Step 2: Heuristic mapping based on natural language meaning
    if not target_field:
        clean_meaning = re.sub(r"[_\-]+", " ", meaning_lower)
        if re.search(r"\b(assign|assigned|assignee|owned|owner|patented by|patent by|company|applicant)\b", clean_meaning):
            target_field = "current_assignee_normalized"
        elif re.search(r"\b(active|alive|expired|dead)\b", clean_meaning):
            target_field = "legal_state"
        elif re.search(r"\b(pending|abandoned|granted|grant|ceased)\b", clean_meaning):
            target_field = "legal_status"
        elif re.search(r"\b(invented by|inventor|inventors|author)\b", clean_meaning):
            target_field = "inventor"
        elif "publication" in clean_meaning or "published" in clean_meaning:
            if _is_year_or_number(val_str) or "year" in clean_meaning:
                target_field = "publication_year"
            else:
                target_field = "publication_country_code"
        elif "priority" in clean_meaning:
            if _is_year_or_number(val_str) or "year" in clean_meaning:
                target_field = "earliest_priority_year"
            else:
                target_field = "priority_country"
        elif re.search(r"\b(filed|filing|application|from|country|in the|in)\b", clean_meaning):
            if _is_year_or_number(val_str) or "year" in clean_meaning:
                target_field = "application_year"
            else:
                target_field = "application_country"
        elif normalize_country(val_str) is not None:
            target_field = "application_country"

    if not target_field or target_field not in FIELD_MAPPING:
        return None

    spec = FIELD_MAPPING[target_field]
    field_type = spec["type"]
    allowed_ops = spec["operators"]

    # Step 3: Determine operator
    operator = "equals"
    if field_type in ("number", "array_number"):
        if any(w in meaning_lower for w in ["after", "later", "gt", "greater"]):
            operator = "gt"
        elif any(w in meaning_lower for w in ["before", "earlier", "lt", "less"]):
            operator = "lt"
        elif any(w in meaning_lower for w in ["since", "from"]):
            operator = "gte"
        elif any(w in meaning_lower for w in ["until"]):
            operator = "lte"
        else:
            operator = "equals"
    elif target_field in _COUNTRY_FIELDS:
        # Country fields use exact ISO code matching, not substring contains
        operator = "equals"
    elif field_type in ("array_string", "string"):
        if "contains" in allowed_ops:
            operator = "contains"
        else:
            operator = "equals"

    if operator not in allowed_ops:
        operator = allowed_ops[0]


    # Step 4: Value normalization and validation
    norm_val: Any = None

    if target_field in ("application_country", "publication_country_code", "priority_country"):
        norm_val = normalize_country(val_str)
        if not norm_val:
            return None
    elif target_field in ("current_assignee_normalized", "current_assignee_standardized", "original_assignee", "applicant_first_organization"):
        norm_val = normalize_org_name(val_str)
        if not norm_val:
            return None
    elif target_field == "legal_state":
        v_low = val_str.lower()
        if v_low in ("active", "alive"):
            norm_val = "Alive"
        elif v_low in ("expired", "dead"):
            norm_val = "Dead"
        elif val_str in ("Alive", "Dead"):
            norm_val = val_str
        else:
            return None
    elif target_field == "legal_status":
        v_low = val_str.lower()
        if v_low in ("pending", "filed"):
            norm_val = "Filed"
        elif v_low in ("granted", "grant"):
            norm_val = "Granted"
        elif v_low in ("abandoned", "ceased"):
            norm_val = "Ceased"
        elif val_str in ("Filed", "Granted", "Ceased"):
            norm_val = val_str
        else:
            return None
    elif target_field == "publication_type":
        v_low = val_str.lower()
        if "grant" in v_low:
            norm_val = "Grant"
        elif "app" in v_low:
            norm_val = "Application"
        elif val_str in ("Grant", "Application"):
            norm_val = val_str
        else:
            return None
    elif field_type in ("number", "array_number"):
        try:
            num = float(val_str)
            norm_val = int(num) if num.is_integer() else num
        except (ValueError, TypeError):
            return None
    else:
        norm_val = val_str

    return MetadataFilter(field=target_field, operator=operator, value=norm_val)


class QueryUnderstanding:
    """
    LLM-based Query Understanding parser with FIELD_MAPPING allowlist resolution.
    Falls back gracefully to rule-based parsing if local LLM is unavailable.
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
                self.model_name, trust_remote_code=True
            )
            print("Local LLM loaded successfully.\n")
        except Exception as e:
            print(f"[QueryUnderstanding] Warning: Local LLM ({self.model_name}) not loaded: {e}. Using rule-based query understanding parser.")
            self._llm_model = None
            self._llm_tokenizer = None

    def _call_llm(self, query: str) -> dict | None:
        self._init_llm()
        if self._llm_model is None or self._llm_tokenizer is None:
            return None

        prompt = f"""<|im_start|>system
You are the query-understanding component of a patent semantic search engine.

Separate the user's natural-language query into:

1. semantic_query
   The actual invention/topic that should be searched semantically.

2. candidate_filters
   Metadata constraints expressed only using:
   - meaning
   - value

IMPORTANT:

- Never output database field names.
- Never output FIELD_MAPPING keys.
- Never invent metadata fields.
- Never output database schema names.
- Never decide which database field a meaning corresponds to.
- The application will resolve meanings against an official allowlist.
- If a phrase is not clearly a metadata constraint, keep it in semantic_query.
- Preserve user values.
- Extract multiple filters when appropriate.
- Remove recognized metadata constraints from semantic_query.
- Return valid JSON only.
- No Markdown.
- No explanation.

Example:

User:
bottle designs patented by Coca Cola in the US

Output:
{{"semantic_query": "bottle designs", "candidate_filters": [{{"meaning": "patented by", "value": "Coca Cola"}}, {{"meaning": "in the US", "value": "US"}}]}}

Example:

User:
bottle design patents published in Japan after 2018

Output:
{{"semantic_query": "bottle design", "candidate_filters": [{{"meaning": "published in Japan", "value": "Japan"}}, {{"meaning": "published after", "value": "2018"}}]}}

Example:

User:
bottle design

Output:
{{"semantic_query": "bottle design", "candidate_filters": []}}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
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

        # Step 1: Attempt local LLM query understanding
        llm_res = self._call_llm(query)

        if llm_res and isinstance(llm_res, dict) and "semantic_query" in llm_res:
            semantic_query = str(llm_res.get("semantic_query", query)).strip()
            if not semantic_query:
                semantic_query = query

            raw_candidates = llm_res.get("candidate_filters", [])
            candidate_filters: list[CandidateFilter] = []
            metadata_filters: list[MetadataFilter] = []

            for item in raw_candidates:
                if isinstance(item, dict) and "meaning" in item and "value" in item:
                    meaning = str(item["meaning"]).strip()
                    value = item["value"]
                    cf = CandidateFilter(meaning=meaning, value=value)
                    candidate_filters.append(cf)

                    resolved = resolve_candidate_filter(meaning, value)
                    if resolved and resolved.field in FIELD_MAPPING:
                        metadata_filters.append(resolved)

            return ParsedQuery(
                original_query=original,
                semantic_query=semantic_query,
                candidate_filters=candidate_filters,
                metadata_filters=metadata_filters,
            )

        # Fallback: Rule-Based Parser
        return self.parse_rule_based(query)

    def parse_rule_based(self, query: str) -> ParsedQuery:
        original = query
        query = query.strip()

        if not query:
            return ParsedQuery(original_query=original, semantic_query=original, metadata_filters=[])

        consumed: list[tuple[int, int]] = []
        filters: list[MetadataFilter] = []

        # Phase 1: curated aliases
        for field_name, spec in FIELD_MAPPING.items():
            if spec["type"] == "enum":
                continue
            for result in self._extract_field(query, field_name, spec, consumed):
                metadata_filter, span = result
                filters.append(metadata_filter)
                consumed.append(span)

        # Phase 2: enum bare-value triggers
        for field_name, spec in FIELD_MAPPING.items():
            if spec["type"] != "enum":
                continue
            result = self._extract_enum(query, field_name, spec, consumed)
            if result is not None:
                metadata_filter, span = result
                filters.append(metadata_filter)
                consumed.append(span)

        # Phase 3: country idioms
        if not any(f.field == "application_country" for f in filters):
            result = self._extract_leading_country_idiom(query, consumed)
            if result is not None:
                metadata_filter, country_span, patent_span = result
                filters.append(metadata_filter)
                consumed.append(country_span)
                consumed.append(patent_span)

        if not any(f.field == "application_country" for f in filters):
            result = self._extract_trailing_country_idiom(query, consumed)
            if result is not None:
                metadata_filter, span = result
                filters.append(metadata_filter)
                consumed.append(span)

        # Phase 4: bare org name
        if not any(f.field in ("current_assignee_normalized", "original_assignee", "applicant_first_organization") for f in filters):
            result = self._extract_bare_org_name(query, consumed)
            if result is not None:
                metadata_filter, span = result
                filters.append(metadata_filter)
                consumed.append(span)

        consumed = self._absorb_preceding_patent_word(query, consumed)
        semantic_query = self._build_semantic_query(query, consumed)

        candidate_filters = [
            CandidateFilter(meaning=f.field, value=f.value)
            for f in filters
        ]

        return ParsedQuery(
            original_query=original,
            semantic_query=semantic_query,
            candidate_filters=candidate_filters,
            metadata_filters=filters,
        )

    # ------------------------------------------------------------
    # Rule-Based Helper Methods (Preserved)
    # ------------------------------------------------------------

    def _extract_field(self, query, field_name, spec, consumed):
        if not spec["aliases"]:
            return

        is_country = field_name in _COUNTRY_FIELDS
        is_number = spec["type"] in ("number", "array_number")

        for alias in spec["aliases"]:
            alias_span = self._find_alias(query, alias, consumed)
            if alias_span is None:
                continue

            alias_start, alias_end = alias_span

            if is_number:
                extracted = self._extract_number_value(query, alias_end)
            elif is_country:
                extracted = self._extract_country_value(query, alias_end)
            else:
                extracted = self._extract_text_value(query, alias_end, consumed)

            if extracted is None:
                continue

            value, value_end, operator = extracted

            if self._overlaps((alias_end, value_end), consumed):
                continue

            yield (
                MetadataFilter(field=field_name, operator=operator, value=value),
                (alias_start, value_end),
            )

            if is_country:
                chained = self._extract_number_value(query, value_end)
                if chained is not None:
                    year_value, year_end, year_operator = chained
                    if not self._overlaps((value_end, year_end), consumed):
                        yield (
                            MetadataFilter(
                                field=_COUNTRY_TO_YEAR_FIELD[field_name],
                                operator=year_operator,
                                value=year_value,
                            ),
                            (value_end, year_end),
                        )

            return

    def _extract_enum(self, query, field_name, spec, consumed):
        lower_query = query.lower()

        for value in spec["values"]:
            triggers = {value.lower()}
            triggers.update(t.lower() for t in _ENUM_SYNONYMS.get(field_name, {}).get(value, []))

            for trigger in triggers:
                match = re.search(rf"\b{re.escape(trigger)}\b", lower_query)
                if match is None:
                    continue
                span = match.span()
                if self._overlaps(span, consumed):
                    continue
                return MetadataFilter(field=field_name, operator="equals", value=value), span

        return None

    def _find_alias(self, query, alias, consumed):
        match = re.search(rf"\b{re.escape(alias)}\b", query, re.IGNORECASE)
        if match is None:
            return None
        if self._overlaps(match.span(), consumed):
            return None
        return match.span()

    def _extract_country_value(self, query, start):
        window = query[start:start + 60]
        stop = _COMPARATOR_RE.search(window)
        if stop:
            window = window[:stop.start()]

        found = find_country_in_text(window)
        if found is None:
            return None

        code, end = found
        return code, start + end, "equals"

    def _extract_number_value(self, query, start):
        window = query[start:start + 30]
        match = _NUMBER_VALUE_RE.match(window)
        if match is None:
            return None

        comparator = (match.group(1) or "").lower()
        operator = _COMPARATOR_OPERATORS.get(comparator, "equals")

        value = float(match.group(2))
        if value.is_integer():
            value = int(value)

        return value, start + match.end(), operator

    def _extract_text_value(self, query, start, consumed):
        remainder = query[start:]

        stop_pos = len(remainder)
        for stopword in _TEXT_CAPTURE_STOPWORDS:
            match = re.search(rf"\b{re.escape(stopword)}\b", remainder, re.IGNORECASE)
            if match and match.start() < stop_pos:
                stop_pos = match.start()

        for other_start, other_end in consumed:
            if other_start > start and (other_start - start) < stop_pos:
                stop_pos = other_start - start

        raw = remainder[:stop_pos]
        raw = re.sub(r"^[\s,:\-]+|[\s,:.\-?!]+$", "", raw)
        raw = re.sub(r"^(the|a|an)\s+", "", raw, flags=re.IGNORECASE)

        value = normalize_org_name(raw)
        if not value:
            return None

        return value, start + stop_pos, "contains"

    def _extract_leading_country_idiom(self, query, consumed):
        first_word_match = re.match(r"^([A-Za-z][A-Za-z.]*)", query)
        if not first_word_match:
            return None

        trimmed_end = query.rstrip(" ?.!")
        last_word_match = re.search(r"([A-Za-z]+)$", trimmed_end)
        if not last_word_match or last_word_match.group(1).lower() not in ("patent", "patents"):
            return None

        code = normalize_country(first_word_match.group(1))
        if code is None:
            return None

        country_span = (0, first_word_match.end())
        patent_span = (last_word_match.start(), len(trimmed_end))

        if self._overlaps(country_span, consumed) or self._overlaps(patent_span, consumed):
            return None

        return (
            MetadataFilter(field="application_country", operator="equals", value=code),
            country_span,
            patent_span,
        )

    def _extract_trailing_country_idiom(self, query, consumed):
        trimmed_end = query.rstrip(" ?.!")
        match = _TRAILING_COUNTRY_RE.search(trimmed_end)
        if not match:
            return None

        code = normalize_country(match.group(1))
        if code is None:
            return None

        span = (match.start(), len(trimmed_end))
        if self._overlaps(span, consumed):
            return None

        return (
            MetadataFilter(field="application_country", operator="equals", value=code),
            span,
        )

    def _extract_bare_org_name(self, query, consumed):
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
            span = match.span()
            if self._overlaps(span, consumed):
                continue
            if normalize_country(match.group(1)) is not None:
                continue

            value = normalize_org_name(match.group(1))
            if not value:
                continue

            return (
                MetadataFilter(field="current_assignee_normalized", operator="contains", value=value),
                span,
            )

        return None

    def _absorb_preceding_patent_word(self, query, consumed):
        absorbed = []
        for start, end in consumed:
            before = query[:start]
            before_stripped = before.rstrip()

            match = re.search(r"\b(patents?|applications?)$", before_stripped, re.IGNORECASE)
            if match:
                new_start = match.start()
                absorbed.append((new_start, end))
            else:
                absorbed.append((start, end))

        return absorbed

    def _build_semantic_query(self, query, consumed):
        ordered = sorted(consumed)
        pieces = []
        cursor = 0

        for start, end in ordered:
            start = max(start, cursor)
            if start > cursor:
                pieces.append(query[cursor:start])
            cursor = max(cursor, end)

        pieces.append(query[cursor:])

        residual = " ".join(" ".join(pieces).split())
        tokens = residual.split(" ") if residual else []

        while tokens and self._strip_token(tokens[0]) in _LEADING_FILLER:
            tokens.pop(0)

        cleaned = " ".join(tokens).strip()

        if consumed and cleaned:
            trailing_match = _TRAILING_PATENT_WORD_RE.search(cleaned)
            if trailing_match and trailing_match.start() > 0:
                candidate = cleaned[:trailing_match.start()].rstrip()
                if candidate:
                    cleaned = candidate

        if not cleaned or not self._strip_token(cleaned):
            return query

        return cleaned

    @staticmethod
    def _strip_token(token: str) -> str:
        return re.sub(r"[^a-zA-Z]", "", token).lower()

    @staticmethod
    def _overlaps(span, consumed) -> bool:
        start, end = span
        return any(start < c_end and end > c_start for c_start, c_end in consumed)

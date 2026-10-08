import re
from typing import Any, ClassVar

from app.models.parsed_query import (
    ConceptAttribute,
    MetadataFilter,
    ParsedQuery,
    SemanticRelationship,
)
from metadata_fields import METADATA_FIELD_CODES


class QueryNormalizer:
    
    def __init__(self):
        self._build_metadata_index()

    def _build_metadata_index(self):
        # 1. Direct valid canonical codes (e.g. 'AS_EN', 'PY', 'CPC')
        self.canonical_codes: set[str] = set(METADATA_FIELD_CODES.keys())

        # 2. Lowercase description to canonical code lookup
        self.desc_to_code: dict[str, str] = {}
        for code, desc in METADATA_FIELD_CODES.items():
            clean_desc = desc.lower().replace("\n", " ").strip()
            self.desc_to_code[clean_desc] = code

        # 3. Common natural-language patent terminology aliases
        self.alias_to_code: dict[str, str] = {
            # Assignee / Applicant / Company
            "assignee": "AS_EN",
            "assignee name": "AS_EN",
            "assignee_name": "AS_EN",
            "applicant": "AP_EN",
            "applicant name": "AP_EN",
            "company": "AS_EN",
            "owner": "CAS_EN",
            "organization": "AP_EN",
            "current assignee": "CA_EN",
            "original assignee": "AO_EN",
            # Inventor
            "inventor": "IN_EN",
            "inventor name": "IN_EN",
            "author": "IN_EN",
            "first inventor": "INF_EN",
            # Dates and Years
            "publication year": "PY",
            "publication_year": "PY",
            "published year": "PY",
            "pub year": "PY",
            "year": "PY",
            "publication date": "PD",
            "publication_date": "PD",
            "published date": "PD",
            "pub date": "PD",
            "date": "PD",
            "filing year": "AY",
            "filing_year": "AY",
            "filed year": "AY",
            "application year": "AY",
            "application_year": "AY",
            "app year": "AY",
            "filing date": "AD",
            "filing_date": "AD",
            "filed date": "AD",
            "application date": "AD",
            "application_date": "AD",
            "app date": "AD",
            "priority date": "PRD",
            "priority year": "PRY",
            "earliest priority date": "EPRD",
            # Country / Jurisdiction
            "country": "PNC",
            "country code": "PNC",
            "publication country": "PNC",
            "jurisdiction": "PNC",
            "application country": "AC",
            "priority country": "PRC",
            # Classifications
            "cpc": "CPC",
            "cpc code": "CPC",
            "cpc class": "CPC",
            "cpc classification": "CPC",
            "ipc": "IPC",
            "ipc code": "IPC",
            "ipc class": "IPC",
            "ipc classification": "IPC",
            "us class": "USM",
            # Content sections
            "title": "TA",
            "abstract": "AB_EN",
            "claim": "CL_EN",
            "claims": "CL_EN",
            "independent claim": "ICL_EN",
            "independent claims": "ICL_EN",
            # Numbers and Status
            "publication number": "PN_D",
            "patent number": "PN_D",
            "application number": "AN_D",
            "legal status": "LST",
            "legal state": "ALD",
        }

        # Country name to ISO 2-letter code mapping
        self.country_map: dict[str, str] = {
            "united states": "US",
            "united states of america": "US",
            "usa": "US",
            "us": "US",
            "japan": "JP",
            "jp": "JP",
            "china": "CN",
            "cn": "CN",
            "europe": "EP",
            "european patent office": "EP",
            "ep": "EP",
            "germany": "DE",
            "de": "DE",
            "korea": "KR",
            "south korea": "KR",
            "kr": "KR",
            "united kingdom": "GB",
            "great britain": "GB",
            "uk": "GB",
            "gb": "GB",
            "france": "FR",
            "fr": "FR",
            "canada": "CA",
            "ca": "CA",
            "taiwan": "TW",
            "tw": "TW",
            "wipo": "WO",
            "world intellectual property organization": "WO",
            "pct": "WO",
            "wo": "WO",
            "australia": "AU",
            "au": "AU",
            "india": "IN",
        }

        # Standard operator mapping
        self.operator_map: dict[str, str] = {
            "=": "==",
            "==": "==",
            "eq": "==",
            "equals": "==",
            "is": "==",
            "exact": "==",
            "!=": "!=",
            "ne": "!=",
            "neq": "!=",
            "not equals": "!=",
            "is not": "!=",
            ">": ">",
            "gt": ">",
            "after": ">",
            "greater than": ">",
            "more than": ">",
            ">=": ">=",
            "gte": ">=",
            "on or after": ">=",
            "greater than or equal to": ">=",
            "<": "<",
            "lt": "<",
            "before": "<",
            "less than": "<",
            "earlier than": "<",
            "<=": "<=",
            "lte": "<=",
            "on or before": "<=",
            "less than or equal to": "<=",
            "contains": "contains",
            "like": "contains",
            "match": "contains",
            "includes": "contains",
            "has": "contains",
            "in": "in",
            "one of": "in",
            "any of": "in",
            "between": "between",
            "range": "between",
        }

    _ALIVE_DEAD_VALUES: ClassVar[set[str]] = {"alive", "dead"}
    _FILED_GRANTED_CEASED_VALUES: ClassVar[set[str]] = {"filed", "granted", "ceased"}

    def _correct_legal_status_field(self, field: str, value: Any) -> str:
        if field not in ("LST", "ALD"):
            return field
        val_lower = value.strip().lower() if isinstance(value, str) else None
        if val_lower in self._ALIVE_DEAD_VALUES:
            return "ALD"
        if val_lower in self._FILED_GRANTED_CEASED_VALUES:
            return "LST"
        return field

    def resolve_metadata_field(self, raw_field: str) -> tuple[str, str | None]:
        if not raw_field:
            return "UNKNOWN", raw_field

        cleaned = raw_field.strip()
        upper_code = cleaned.upper()

        # 1. Exact match with canonical code (e.g. 'AS_EN', 'PY')
        if upper_code in self.canonical_codes:
            return upper_code, cleaned

        lower_desc = cleaned.lower().replace("_", " ").strip()

        # 2. Match against alias map
        if lower_desc in self.alias_to_code:
            return self.alias_to_code[lower_desc], cleaned

        # 3. Match against exact descriptions from METADATA_FIELD_CODES
        if lower_desc in self.desc_to_code:
            return self.desc_to_code[lower_desc], cleaned

        # 4. Partial substring match against descriptions
        for desc, code in self.desc_to_code.items():
            if lower_desc == desc or lower_desc in desc.split(" - ")[0]:
                return code, cleaned

        # 5. Return original un-resolved field cleanly
        return cleaned, cleaned

    def normalize_operator(self, op: str) -> str:
        if not op:
            return "=="
        cleaned = op.strip().lower()
        return self.operator_map.get(cleaned, "==")

    def normalize_value(self, field: str, value: Any) -> Any:
        if isinstance(value, str):
            val_clean = value.strip().strip("'\"")
            # Country code normalization
            if field in ("PNC", "AC", "PRC", "PCTPC"):
                val_lower = val_clean.lower()
                if val_lower in self.country_map:
                    return self.country_map[val_lower]
                return val_clean.upper()
            # Year normalization
            if field in ("PY", "AY", "PRY", "EPRY") and val_clean.isdigit() and len(val_clean) == 4:
                return int(val_clean)

            if field in (
                "CPC",
                "CPCP",
                "CPC12",
                "CPC4",
                "CPC8",
                "IPC",
                "IPC12",
                "IPC4",
                "IPC8",
            ):
                return re.sub(r"[\s/\-.]", "", val_clean).upper()
            return val_clean

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, list):
            return [self.normalize_value(field, v) for v in value]

        return value

    def normalize(self, query: ParsedQuery) -> ParsedQuery:
        original_query = query.original_query.strip()
        semantic_query = (
            query.semantic_query.strip() if query.semantic_query else original_query
        )
        structured_query = (
            query.structured_query.strip() if query.structured_query else semantic_query
        )
        evidence_query = (
            query.evidence_query.strip() if query.evidence_query else semantic_query
        )

        # 1. Deduplicate concepts (case-insensitive, preserving order)
        seen_concepts: set[str] = set()
        deduped_concepts: list[str] = []
        for c in query.concepts:
            c_clean = c.strip()
            if not c_clean:
                continue
            c_key = c_clean.lower()
            if c_key not in seen_concepts:
                seen_concepts.add(c_key)
                deduped_concepts.append(c_clean)

        # 2. Deduplicate and clean relationships
        seen_rels: set[tuple[str, str, str]] = set()
        deduped_rels: list[SemanticRelationship] = []
        for r in query.relationships:
            subj = r.subject.strip()
            rel = r.relation.strip()
            obj = r.object.strip()
            if not (subj and rel and obj):
                continue
            r_key = (subj.lower(), rel.lower(), obj.lower())
            if r_key not in seen_rels:
                seen_rels.add(r_key)
                deduped_rels.append(
                    SemanticRelationship(
                        subject=subj,
                        relation=rel,
                        object=obj,
                        context=r.context.strip() if r.context else None,
                    )
                )

        # 3. Deduplicate and clean attributes
        seen_attrs: set[tuple[str, str, str]] = set()
        deduped_attrs: list[ConceptAttribute] = []
        for a in query.attributes:
            c = a.concept.strip()
            n = a.name.strip()
            v = a.value.strip()
            if not (c and n and v):
                continue
            a_key = (c.lower(), n.lower(), v.lower())
            if a_key not in seen_attrs:
                seen_attrs.add(a_key)
                deduped_attrs.append(ConceptAttribute(concept=c, name=n, value=v))

        # 4. Deduplicate string lists (requirements, constraints, exclusions)
        deduped_reqs = self._dedupe_str_list(query.requirements)
        deduped_constraints = self._dedupe_str_list(query.constraints)
        deduped_exclusions = self._dedupe_str_list(query.exclusions)

        # 5. Resolve and deduplicate metadata filters
        seen_filters: set[tuple[str, str, str]] = set()
        deduped_filters: list[MetadataFilter] = []
        for f in query.metadata_filters:
            canonical_field, raw_field = self.resolve_metadata_field(
                f.field or f.raw_field or ""
            )
            op = self.normalize_operator(f.operator)
            val = self.normalize_value(canonical_field, f.value)
            canonical_field = self._correct_legal_status_field(canonical_field, val)
            f_key = (canonical_field, op, str(val).lower())
            if f_key not in seen_filters:
                seen_filters.add(f_key)
                deduped_filters.append(
                    MetadataFilter(
                        field=canonical_field,
                        operator=op,
                        value=val,
                        raw_field=raw_field or f.raw_field,
                    )
                )

        # 6. Safety check for is_metadata_only
        is_metadata_only = query.is_metadata_only
        # If there are no concepts, relationships, or requirements, and only metadata filters exist, it is metadata only
        if (
            len(deduped_filters) > 0
            and len(deduped_concepts) == 0
            and len(deduped_rels) == 0
        ):
            is_metadata_only = True
        elif len(deduped_concepts) > 0 or len(deduped_rels) > 0:
            # Has technical concepts or relationships
            is_metadata_only = False

        return ParsedQuery(
            original_query=original_query,
            semantic_query=semantic_query,
            structured_query=structured_query,
            evidence_query=evidence_query,
            concepts=deduped_concepts,
            relationships=deduped_rels,
            attributes=deduped_attrs,
            requirements=deduped_reqs,
            constraints=deduped_constraints,
            exclusions=deduped_exclusions,
            metadata_filters=deduped_filters,
            is_metadata_only=is_metadata_only,
        )

    @staticmethod
    def _dedupe_str_list(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            clean = item.strip()
            if not clean:
                continue
            key = clean.lower()
            if key not in seen:
                seen.add(key)
                result.append(clean)
        return result

"""
Field Mapping — Curated Metadata Filter Allowlist

The single source of truth for which metadata fields QueryUnderstanding
is allowed to filter on. A filter's field is only ever accepted if it's
a key in FIELD_MAPPING - this is the allowlist that stops the parser
(or LLM-based query understanding) from inventing unsupported fields.

IMPORTANT: ``real_key`` must be exactly the key used in stored patent
metadata JSON (verified by inspecting patents-processed/*.json), NOT
the formal name from METADATA_FIELD_CODES which can differ (e.g. the
formal code CAS_EN maps to "Current Assignee Standardized English" but
the actual stored key is "Current Assignee Standardized").

To add new filter keys in the future:
1. Inspect the actual stored metadata JSON to find the exact key name.
2. Add the corresponding config block to FIELD_MAPPING below.
"""

from __future__ import annotations

# ---- Actual stored metadata key names (authoritative) ----
# Verified by inspecting patents-processed/*.json files.
# These are the exact dictionary keys used in the patent metadata
# payloads stored in Qdrant's "patents" collection.

_REAL_KEYS = {
    "AC":       "Application Country",
    "AY":       "Application Year",
    "PNC":      "Publication Country Code",
    "PY":       "Publication Year",
    "PT":       "Publication Type",
    "APT":      "Applicant Type",
    "APFO_EN":  "Applicant First Organization",
    "IN_EN":    "Inventor",
    "INF_EN":   "Inventor First",
    "PRC":      "Priority Country",
    "EPRY":     "Earliest Priority Year",
    "PRY":      "Priority Year",
    "CAS_EN":   "Current Assignee Standardized",
    "LST":      "Legal Status (Filed/Granted/Ceased)",
    "ALD":      "Legal State\n(Alive/Dead)",
    "AAPO":     "Assignee/Applicant (Original) with Address",
    "AAPS":     "Assignee/Applicant (Standardized) with Address",
    # The following keys may appear in richer metadata exports.
    # If they are not present in a given patent's JSON, the filter
    # simply won't match (FilterEngine returns False for missing keys).
    "CAN_EN":   "Current Assignee Normalized",
    "CPC":      "CPC",
    "CPCP":     "CPC Primary",
    "CPC12":    "CPC - 12 Digit",
    "CPC4":     "CPC - 8 Digit",
    "CPC8":     "CPC - 4 Digit",
    "CPCV":     "CPC - Version",
    "CPCO":     "CPC - Assigning Office",
    "IPC":      "IPC",
    "IPC12":    "IPC - 12 Digit",
    "IPC8":     "IPC - 8 Digit",
    "IPC4":     "IPC - 4 Digit",
    "IPCRV":    "IPCR Version",
}


FIELD_MAPPING: dict[str, dict] = {
    "application_country": {
        "code": "AC",
        "real_key": _REAL_KEYS["AC"],
        "type": "string",
        "operators": ["equals", "contains"],
        "aliases": [
            "application country",
            "country of application",
            "country filed in",
            "filed in",
            "applied in",
            "from",
        ],
    },
    "application_year": {
        "code": "AY",
        "real_key": _REAL_KEYS["AY"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
        "aliases": [
            "application year",
            "filing year",
            "filed in",
        ],
    },
    "publication_country_code": {
        "code": "PNC",
        "real_key": _REAL_KEYS["PNC"],
        "type": "string",
        "operators": ["equals", "contains"],
        "aliases": [
            "publication country",
            "country published in",
            "published in",
        ],
    },
    "publication_year": {
        "code": "PY",
        "real_key": _REAL_KEYS["PY"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
        "aliases": [
            "publication year",
            "published in",
        ],
    },
    "publication_type": {
        "code": "PT",
        "real_key": _REAL_KEYS["PT"],
        "type": "enum",
        "operators": ["equals"],
        "values": ["Grant", "Application"],
        "aliases": ["publication type"],
    },
    "priority_country": {
        "code": "PRC",
        "real_key": _REAL_KEYS["PRC"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [
            "priority country",
            "priority in",
        ],
    },
    "priority_year": {
        "code": "PRY",
        "real_key": _REAL_KEYS["PRY"],
        "type": "array_number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
        "aliases": [
            "priority year",
        ],
    },
    "earliest_priority_year": {
        "code": "EPRY",
        "real_key": _REAL_KEYS["EPRY"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
        "aliases": [
            "earliest priority year",
        ],
    },
    "current_assignee_normalized": {
        "code": "CAN_EN",
        "real_key": _REAL_KEYS["CAN_EN"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [
            "currently owned by",
            "current assignee",
            "assigned to",
            "owned by",
        ],
    },
    "current_assignee_standardized": {
        "code": "CAS_EN",
        "real_key": _REAL_KEYS["CAS_EN"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [],
    },
    "original_assignee": {
        "code": "AAPO",
        "real_key": _REAL_KEYS["AAPO"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [
            "originally assigned to",
            "original assignee",
        ],
    },
    "applicant_first_organization": {
        "code": "APFO_EN",
        "real_key": _REAL_KEYS["APFO_EN"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [
            "applicant",
            "filed by",
        ],
    },
    "applicant_type": {
        "code": "APT",
        "real_key": _REAL_KEYS["APT"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [],
    },
    "inventor": {
        "code": "IN_EN",
        "real_key": _REAL_KEYS["IN_EN"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": [
            "invented by",
            "inventor",
        ],
    },
    "legal_status": {
        "code": "LST",
        "real_key": _REAL_KEYS["LST"],
        "type": "enum",
        "operators": ["equals"],
        "values": ["Filed", "Granted", "Ceased"],
        "aliases": ["legal status"],
    },
    "legal_state": {
        "code": "ALD",
        "real_key": _REAL_KEYS["ALD"],
        "type": "enum",
        "operators": ["equals"],
        "values": ["Alive", "Dead"],
        "aliases": ["legal state"],
    },
    "cpc_primary": {
        "code": "CPCP",
        "real_key": _REAL_KEYS["CPCP"],
        "type": "string",
        "operators": ["equals", "contains"],
        "aliases": ["cpc primary"],
    },
    "cpc": {
        "code": "CPC",
        "real_key": _REAL_KEYS["CPC"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["cpc", "cpc class", "cpc classification"],
    },
    "cpc_12_digit": {
        "code": "CPC12",
        "real_key": _REAL_KEYS["CPC12"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["cpc 12 digit"],
    },
    "cpc_4_digit": {
        "code": "CPC4",
        "real_key": _REAL_KEYS["CPC4"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["cpc 4 digit", "cpc section"],
    },
    "cpc_8_digit": {
        "code": "CPC8",
        "real_key": _REAL_KEYS["CPC8"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["cpc 8 digit", "cpc subclass"],
    },
    "cpc_version": {
        "code": "CPCV",
        "real_key": _REAL_KEYS["CPCV"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["cpc version"],
    },
    "cpc_assigning_office": {
        "code": "CPCO",
        "real_key": _REAL_KEYS["CPCO"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["cpc assigning office"],
    },
    "ipc": {
        "code": "IPC",
        "real_key": _REAL_KEYS["IPC"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["ipc", "ipc class", "ipc classification"],
    },
    "ipc_12_digit": {
        "code": "IPC12",
        "real_key": _REAL_KEYS["IPC12"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["ipc 12 digit"],
    },
    "ipc_8_digit": {
        "code": "IPC8",
        "real_key": _REAL_KEYS["IPC8"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["ipc 8 digit"],
    },
    "ipc_4_digit": {
        "code": "IPC4",
        "real_key": _REAL_KEYS["IPC4"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["ipc 4 digit"],
    },
    "ipcr_version": {
        "code": "IPCRV",
        "real_key": _REAL_KEYS["IPCRV"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["ipcr version"],
    },
    "inventor_first": {
        "code": "INF_EN",
        "real_key": _REAL_KEYS["INF_EN"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["first inventor", "primary inventor"],
    },
    "assignee_applicant_original_with_address": {
        "code": "AAPO",
        "real_key": _REAL_KEYS["AAPO"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["original assignee with address", "original applicant with address"],
    },
    "assignee_applicant_standardized_with_address": {
        "code": "AAPS",
        "real_key": _REAL_KEYS["AAPS"],
        "type": "array_string",
        "operators": ["contains"],
        "aliases": ["standardized assignee with address", "standardized applicant with address"],
    },
}

_UNSAFE_KEY_CHARS = set(" \n()/")

QDRANT_UNSAFE_FIELDS = frozenset(
    field for field, spec in FIELD_MAPPING.items()
    if _UNSAFE_KEY_CHARS & set(spec["real_key"])
)

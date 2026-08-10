"""
Field Mapping — Curated Metadata Filter Allowlist

The single source of truth for which metadata fields QueryUnderstanding
is allowed to filter on.

IMPORTANT:
`real_key` must exactly match the key used in the stored patent
metadata JSON.

The flow is:

    LLM field code
        ↓
    CODE_TO_FIELD
        ↓
    FIELD_MAPPING
        ↓
    real_key
        ↓
    actual patent metadata JSON
"""

from __future__ import annotations

# ==============================================================
# Actual stored metadata key names
# ==============================================================

_REAL_KEYS = {
    "AC": "Application Country",
    "AD": "Application Date",
    "AY": "Application Year",
    "SC": "Us Application Number with Series Code",
    "PNC": "Publication Country Code",
    "PD": "Publication Date",
    "PY": "Publication Year",
    "PT": "Publication Type",
    "PKC": "Publication Kind Code",
    "ACC": "Assignee Country",
    "AO_EN": "Original Assignee",
    "AS_EN": "Assignee Standardized",
    "CAS_EN": "Current Assignee Standardized",
    "CAN_EN": "Current Assignee Normalized",
    "APT": "Applicant Type",
    "APFO_EN": "Applicant First Organization",
    "IN_EN": "Inventor",
    "INF_EN": "Inventor First",
    "AG_EN": "Attorney/Agent",
    "PEX": "Primary Examiner",
    "AEX": "Assistant Examiner",
    "PRC": "Priority Country",
    "PRD": "Priority Date",
    "EPRD": "Earliest Priority Date",
    "EPRY": "Earliest Priority Year",
    "PRY": "Priority Year",
    "CPC": "CPC",
    "CPCP": "CPC Primary",
    "CPCD": "CPC Divided",
    "CPC12": "CPC - 12 Digit",
    "CPC4": "CPC - 8 Digit",
    "CPC8": "CPC - 4 Digit",
    "CPCV": "CPC - Version",
    "CPCO": "CPC - Assigning Office",
    "IPC": "IPC",
    "IPCD": "IPC Divided",
    "IPC12": "IPC - 12 Digit",
    "IPC8": "IPC - 8 Digit",
    "IPC4": "IPC - 4 Digit",
    "IPCR": "IPCR",
    "IPCRD": "IPCR Divided",
    "IPCR12": "IPCR - 12 Digit",
    "IPCR8": "IPCR - 8 Digit",
    "IPCR4": "IPCR - 4 Digit",
    "IPCRV": "IPCR Version",
    "LST": "Legal Status (Filed/Granted/Ceased)",
    "ALD": "Legal State\n(Alive/Dead)",
    "AAPO": "Assignee/Applicant (Original) with Address",
    "AAPN": "Assignee/Applicant (Normalized) with Address",
    "AAPS": "Assignee/Applicant (Standardized) with Address",
    "TI_EN": "Title English",
    "CLN": "Claims (N)",
    "PTS": (
        "Patent type "
        "(Utility, Design, Plant, Reissue, Defensive Publication, "
        "Statutory Invention Registration)"
    ),
    "DST": "Docdb Status",
    "CFID": "Complete Family ID",
    "DFID": "Domestic Family ID",
    "EFID": "Extended Family ID",
    "MFID": "Main Family ID",
    "SFID": "Simple Family ID",
    "ED": "Expiry Date (C)",
    "EDN": "Expiry Date (N)",
    "LD": "Lapse Date (C)",
    "LDN": "Lapse Date (N)",
    "RL": "Estimated Remaining Life (Current Date- Expiry Date)",
    "USMS": "US Maintenance Status",
    "GOI": "Government Interest",
    "PL": "Publication Language",
    "EXP": "Examiner",
    "DC": "Domestic Classification",
    "DCD": "Domestic Classification Divided",
    "IPCM": "IPC Main Classification",
    "PCL": "IPC/CPC",
    "PCLD": "IPCD/CPCD",
}


# ==============================================================
# FIELD MAPPING
# ==============================================================

FIELD_MAPPING: dict[str, dict] = {
    "application_country": {
        "code": "AC",
        "real_key": _REAL_KEYS["AC"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "application_date": {
        "code": "AD",
        "real_key": _REAL_KEYS["AD"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "application_year": {
        "code": "AY",
        "real_key": _REAL_KEYS["AY"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
    },
    "application_number_series_code": {
        "code": "SC",
        "real_key": _REAL_KEYS["SC"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "publication_country_code": {
        "code": "PNC",
        "real_key": _REAL_KEYS["PNC"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "publication_date": {
        "code": "PD",
        "real_key": _REAL_KEYS["PD"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "publication_year": {
        "code": "PY",
        "real_key": _REAL_KEYS["PY"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
    },
    "publication_type": {
        "code": "PT",
        "real_key": _REAL_KEYS["PT"],
        "type": "enum",
        "operators": ["equals"],
        "values": ["Grant", "Application"],
    },
    "publication_kind_code": {
        "code": "PKC",
        "real_key": _REAL_KEYS["PKC"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "assignee_country": {
        "code": "ACC",
        "real_key": _REAL_KEYS["ACC"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "original_assignee": {
        "code": "AO_EN",
        "real_key": _REAL_KEYS["AO_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "assignee_standardized": {
        "code": "AS_EN",
        "real_key": _REAL_KEYS["AS_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "current_assignee_normalized": {
        "code": "CAN_EN",
        "real_key": _REAL_KEYS["CAN_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "current_assignee_standardized": {
        "code": "CAS_EN",
        "real_key": _REAL_KEYS["CAS_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "applicant_type": {
        "code": "APT",
        "real_key": _REAL_KEYS["APT"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "applicant_first_organization": {
        "code": "APFO_EN",
        "real_key": _REAL_KEYS["APFO_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "inventor": {
        "code": "IN_EN",
        "real_key": _REAL_KEYS["IN_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "inventor_first": {
        "code": "INF_EN",
        "real_key": _REAL_KEYS["INF_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "attorney_agent": {
        "code": "AG_EN",
        "real_key": _REAL_KEYS["AG_EN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "primary_examiner": {
        "code": "PEX",
        "real_key": _REAL_KEYS["PEX"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "assistant_examiner": {
        "code": "AEX",
        "real_key": _REAL_KEYS["AEX"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "priority_country": {
        "code": "PRC",
        "real_key": _REAL_KEYS["PRC"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "priority_date": {
        "code": "PRD",
        "real_key": _REAL_KEYS["PRD"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "earliest_priority_date": {
        "code": "EPRD",
        "real_key": _REAL_KEYS["EPRD"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "earliest_priority_year": {
        "code": "EPRY",
        "real_key": _REAL_KEYS["EPRY"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
    },
    "priority_year": {
        "code": "PRY",
        "real_key": _REAL_KEYS["PRY"],
        "type": "array_number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
    },
    "legal_status": {
        "code": "LST",
        "real_key": _REAL_KEYS["LST"],
        "type": "enum",
        "operators": ["equals"],
        "values": ["Filed", "Granted", "Ceased"],
    },
    "legal_state": {
        "code": "ALD",
        "real_key": _REAL_KEYS["ALD"],
        "type": "enum",
        "operators": ["equals"],
        "values": ["Alive", "Dead"],
    },
    "docdb_status": {
        "code": "DST",
        "real_key": _REAL_KEYS["DST"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "cpc_primary": {
        "code": "CPCP",
        "real_key": _REAL_KEYS["CPCP"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "cpc": {
        "code": "CPC",
        "real_key": _REAL_KEYS["CPC"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "cpc_divided": {
        "code": "CPCD",
        "real_key": _REAL_KEYS["CPCD"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "cpc_12_digit": {
        "code": "CPC12",
        "real_key": _REAL_KEYS["CPC12"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "cpc_4_digit": {
        "code": "CPC4",
        "real_key": _REAL_KEYS["CPC4"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "cpc_8_digit": {
        "code": "CPC8",
        "real_key": _REAL_KEYS["CPC8"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "cpc_version": {
        "code": "CPCV",
        "real_key": _REAL_KEYS["CPCV"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "cpc_assigning_office": {
        "code": "CPCO",
        "real_key": _REAL_KEYS["CPCO"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipc": {
        "code": "IPC",
        "real_key": _REAL_KEYS["IPC"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipc_divided": {
        "code": "IPCD",
        "real_key": _REAL_KEYS["IPCD"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipc_12_digit": {
        "code": "IPC12",
        "real_key": _REAL_KEYS["IPC12"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipc_8_digit": {
        "code": "IPC8",
        "real_key": _REAL_KEYS["IPC8"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipc_4_digit": {
        "code": "IPC4",
        "real_key": _REAL_KEYS["IPC4"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcr": {
        "code": "IPCR",
        "real_key": _REAL_KEYS["IPCR"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcr_divided": {
        "code": "IPCRD",
        "real_key": _REAL_KEYS["IPCRD"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcr_12_digit": {
        "code": "IPCR12",
        "real_key": _REAL_KEYS["IPCR12"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcr_8_digit": {
        "code": "IPCR8",
        "real_key": _REAL_KEYS["IPCR8"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcr_4_digit": {
        "code": "IPCR4",
        "real_key": _REAL_KEYS["IPCR4"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcr_version": {
        "code": "IPCRV",
        "real_key": _REAL_KEYS["IPCRV"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "assignee_applicant_original_with_address": {
        "code": "AAPO",
        "real_key": _REAL_KEYS["AAPO"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "assignee_applicant_normalized_with_address": {
        "code": "AAPN",
        "real_key": _REAL_KEYS["AAPN"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "assignee_applicant_standardized_with_address": {
        "code": "AAPS",
        "real_key": _REAL_KEYS["AAPS"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "title_english": {
        "code": "TI_EN",
        "real_key": _REAL_KEYS["TI_EN"],
        "type": "string",
        "operators": ["contains"],
    },
    "claims_count": {
        "code": "CLN",
        "real_key": _REAL_KEYS["CLN"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
    },
    "patent_type": {
        "code": "PTS",
        "real_key": _REAL_KEYS["PTS"],
        "type": "enum",
        "operators": ["equals", "contains"],
        "values": [
            "Utility",
            "Design",
            "Plant",
            "Reissue",
            "Defensive Publication",
            "Statutory Invention Registration",
        ],
    },
    "complete_family_id": {
        "code": "CFID",
        "real_key": _REAL_KEYS["CFID"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "domestic_family_id": {
        "code": "DFID",
        "real_key": _REAL_KEYS["DFID"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "extended_family_id": {
        "code": "EFID",
        "real_key": _REAL_KEYS["EFID"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "main_family_id": {
        "code": "MFID",
        "real_key": _REAL_KEYS["MFID"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "simple_family_id": {
        "code": "SFID",
        "real_key": _REAL_KEYS["SFID"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "expiry_date_current": {
        "code": "ED",
        "real_key": _REAL_KEYS["ED"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "expiry_date_normalized": {
        "code": "EDN",
        "real_key": _REAL_KEYS["EDN"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "lapse_date_current": {
        "code": "LD",
        "real_key": _REAL_KEYS["LD"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "lapse_date_normalized": {
        "code": "LDN",
        "real_key": _REAL_KEYS["LDN"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "estimated_remaining_life": {
        "code": "RL",
        "real_key": _REAL_KEYS["RL"],
        "type": "number",
        "operators": ["equals", "gt", "gte", "lt", "lte"],
    },
    "us_maintenance_status": {
        "code": "USMS",
        "real_key": _REAL_KEYS["USMS"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "government_interest": {
        "code": "GOI",
        "real_key": _REAL_KEYS["GOI"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "publication_language": {
        "code": "PL",
        "real_key": _REAL_KEYS["PL"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "examiner": {
        "code": "EXP",
        "real_key": _REAL_KEYS["EXP"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "domestic_classification": {
        "code": "DC",
        "real_key": _REAL_KEYS["DC"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "domestic_classification_divided": {
        "code": "DCD",
        "real_key": _REAL_KEYS["DCD"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipc_main_classification": {
        "code": "IPCM",
        "real_key": _REAL_KEYS["IPCM"],
        "type": "string",
        "operators": ["equals", "contains"],
    },
    "ipc_cpc": {
        "code": "PCL",
        "real_key": _REAL_KEYS["PCL"],
        "type": "array_string",
        "operators": ["contains"],
    },
    "ipcd_cpcd": {
        "code": "PCLD",
        "real_key": _REAL_KEYS["PCLD"],
        "type": "array_string",
        "operators": ["contains"],
    },
}


_UNSAFE_KEY_CHARS = set(" \n()/")


QDRANT_UNSAFE_FIELDS = frozenset(
    field
    for field, spec in FIELD_MAPPING.items()
    if _UNSAFE_KEY_CHARS & set(spec["real_key"])
)

CODE_TO_FIELD: dict[str, str] = {
    spec["code"]: name for name, spec in FIELD_MAPPING.items()
}

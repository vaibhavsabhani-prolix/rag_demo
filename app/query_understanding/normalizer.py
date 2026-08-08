"""
Value Normalization

Deliberately minimal, non-aggressive normalization between raw query
text and the values FilterEngine compares against stored metadata.
Only mappings that are unambiguous are applied - see each function's
docstring for exactly what it does and does not do.
"""

from __future__ import annotations

import re

# Country name/demonym/abbreviation -> ISO-ish code, as found in this
# project's metadata (Application Country / Publication Country Code /
# Priority Country). Longest keys are tried first so "united states"
# matches before "us".
COUNTRY_ALIASES = {
    "united states of america": "US",
    "united states": "US",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "us": "US",
    "american": "US",
    "china": "CN",
    "chinese": "CN",
    "japan": "JP",
    "japanese": "JP",
    "germany": "DE",
    "german": "DE",
    "south korea": "KR",
    "korea": "KR",
    "korean": "KR",
    "united kingdom": "GB",
    "britain": "GB",
    "british": "GB",
    "uk": "GB",
    "france": "FR",
    "french": "FR",
    "canada": "CA",
    "canadian": "CA",
    "australia": "AU",
    "australian": "AU",
    "taiwan": "TW",
    "taiwanese": "TW",
    "brazil": "BR",
    "brazilian": "BR",
    "spain": "ES",
    "spanish": "ES",
    "russia": "RU",
    "russian": "RU",
    "italy": "IT",
    "italian": "IT",
    "india": "IN",
    "indian": "IN",
    "austria": "AT",
    "austrian": "AT",
    "malaysia": "MY",
    "malaysian": "MY",
    "ukraine": "UA",
    "ukrainian": "UA",
    "european patent office": "EP",
    "epo": "EP",
    "pct": "WO",
}

# Bare two-letter codes accepted case-insensitively (e.g. "US", "CN",
# "JP" as standalone tokens). Codes that double as common English words
# ("in" for India, "it" for Italy, "my" for Malaysia, "us" as a pronoun,
# ...) are deliberately excluded here - they're still reachable through
# their full name/demonym in COUNTRY_ALIASES above, just not from a bare
# ambiguous two-letter token, which would otherwise false-positive
# constantly.
BARE_COUNTRY_CODES = {
    "US", "CN", "JP", "DE", "KR", "GB", "FR", "CA", "AU", "TW", "BR",
    "ES", "RU", "EP", "WO", "AP", "UA",
}

_ORG_PUNCTUATION_RE = re.compile(r"[.,\-]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_country(text: str) -> str | None:
    """
    Resolve *text* to an ISO-ish country code, or None if it doesn't
    match a known country name, demonym, or bare code. Case-insensitive.
    Tries the longest COUNTRY_ALIASES key first so multi-word names
    ("united states") aren't pre-empted by a shorter one.
    """

    stripped = text.strip()
    if not stripped:
        return None

    lower = stripped.lower()

    for name in sorted(COUNTRY_ALIASES, key=len, reverse=True):
        if lower == name:
            return COUNTRY_ALIASES[name]

    if len(stripped) == 2 and stripped.upper() in BARE_COUNTRY_CODES:
        return stripped.upper()

    return None


def normalize_org_name(text: str) -> str:
    """
    Uppercase + strip punctuation (hyphens, periods, commas) + collapse
    whitespace - nothing more. E.g. "Coca-Cola" -> "COCA COLA",
    "coca cola" -> "COCA COLA".

    Deliberately does NOT strip corporate suffixes ("CO", "INC",
    "COMPANY", "LTD", ...): this project's own "*_normalized" metadata
    fields are already vendor-cleaned to bare names (e.g. stored as
    "COCA COLA", not "COCA COLA CO"), so a "contains" match after this
    normalization is enough. Stripping suffixes more aggressively risks
    false positives on legitimately different companies that happen to
    share a suffix.
    """

    cleaned = _ORG_PUNCTUATION_RE.sub(" ", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.upper()

import re
import pycountry

COUNTRY_ALIASES = {
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "american": "US",
    "chinese": "CN",
    "japanese": "JP",
    "german": "DE",
    "korea": "KR",
    "south korea": "KR",
    "korean": "KR",
    "uk": "GB",
    "britain": "GB",
    "british": "GB",
    "england": "GB",
    "english": "GB",
    "french": "FR",
    "canadian": "CA",
    "australian": "AU",
    "taiwanese": "TW",
    "brazilian": "BR",
    "spanish": "ES",
    "russian": "RU",
    "italian": "IT",
    "indian": "IN",
    "austrian": "AT",
    "malaysian": "MY",
    "ukrainian": "UA",
    "swiss": "CH",
    "european patent office": "EP",
    "epo": "EP",
    "pct": "WO",
}

SPECIAL_COUNTRY_CODES = {
    "EP",
    "WO",
    "AP",
}

_ORG_PUNCTUATION_RE = re.compile(r"[.,-]")
_WHITESPACE_RE = re.compile(r"\s+")

def normalize_country(text: str) -> str | None:
    """
    Normalize a country name, country code, or common country
    alias into the canonical country code used by the patent
    metadata.

    Examples:

        Switzerland              -> CH
        switzerland              -> CH
        SWITZERLAND              -> CH
        Swiss                     -> CH
        CH                        -> CH

        China                     -> CN
        china                     -> CN
        Chinese                   -> CN
        CN                        -> CN

        India                     -> IN
        india                     -> IN
        Indian                    -> IN
        IN                        -> IN

        United States             -> US
        USA                       -> US
        American                  -> US
        US                        -> US

    Returns:
        ISO-2 country code / project country code
        or None if the value cannot be resolved.
    """

    if text is None:
        return None

    # Convert to string and clean whitespace.
    value = str(text).strip()

    if not value:
        return None

    value = _WHITESPACE_RE.sub(" ", value)

    lower_value = value.lower()
    upper_value = value.upper()

    # ----------------------------------------------------------
    # 1. Check project-specific aliases
    # ----------------------------------------------------------

    alias = COUNTRY_ALIASES.get(lower_value)

    if alias:
        return alias

    # ----------------------------------------------------------
    # 2. Check special patent-system codes
    # ----------------------------------------------------------

    if upper_value in SPECIAL_COUNTRY_CODES:
        return upper_value

    # ----------------------------------------------------------
    # 3. Check ISO-2 country code dynamically
    # ----------------------------------------------------------

    if len(upper_value) == 2:

        country = pycountry.countries.get(
            alpha_2=upper_value
        )

        if country:
            return country.alpha_2

    # ----------------------------------------------------------
    # 4. Check exact country name
    # ----------------------------------------------------------

    country = pycountry.countries.get(
        name=value
    )

    if country:
        return country.alpha_2

    # ----------------------------------------------------------
    # 5. Case-insensitive country-name lookup
    # ----------------------------------------------------------

    for country in pycountry.countries:

        if country.name.lower() == lower_value:
            return country.alpha_2

    # ----------------------------------------------------------
    # 6. Check official country name
    # ----------------------------------------------------------

    for country in pycountry.countries:

        official_name = getattr(
            country,
            "official_name",
            None,
        )

        if (
            official_name
            and official_name.lower() == lower_value
        ):
            return country.alpha_2

    # ----------------------------------------------------------
    # 7. Check common country name
    # ----------------------------------------------------------

    for country in pycountry.countries:

        common_name = getattr(
            country,
            "common_name",
            None,
        )

        if (
            common_name
            and common_name.lower() == lower_value
        ):
            return country.alpha_2

    # ----------------------------------------------------------
    # 8. Country not recognized
    # ----------------------------------------------------------

    return None


# ==============================================================
# Organization normalization
# ==============================================================

# Stored assignee/applicant "standardized"/"normalized" fields already
# have the legal-entity suffix stripped (that's what "standardized" means
# for a company name) - e.g. "Alios Biopharma Inc" is stored as just
# "ALIOS BIOPHARMA". A query naturally keeps the suffix ("... Inc", "...
# Corp"), and FilterEngine's "contains" is a plain substring check, so the
# longer query value would never match the shorter stored one unless the
# suffix is stripped here too.
_ORG_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "LLC", "LLP", "LP", "PLC",
    "GMBH", "AG", "SA", "SAS", "SARL", "NV", "BV", "KG", "SPA",
    "PTY", "PTE", "OY", "AB", "AS", "KK", "CIE",
}


def normalize_org_name(text: str) -> str:
    """
    Uppercase + strip punctuation (hyphens, periods, commas) + collapse
    whitespace + strip trailing legal-entity suffixes (Inc/Corp/Ltd/...).

    Examples:

        Coca-Cola -> COCA COLA
        coca cola -> COCA COLA
        RANBAXY   -> RANBAXY
        Alios Biopharma, Inc. -> ALIOS BIOPHARMA
        Foo Bar Co Ltd -> FOO BAR
    """

    cleaned = _ORG_PUNCTUATION_RE.sub(" ", text)

    cleaned = _WHITESPACE_RE.sub(
        " ",
        cleaned
    ).strip().upper()

    tokens = cleaned.split(" ") if cleaned else []
    while len(tokens) > 1 and tokens[-1] in _ORG_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)
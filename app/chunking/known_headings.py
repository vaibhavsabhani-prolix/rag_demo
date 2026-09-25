"""
Known section headings for the patent .txt corpus.

Canonical set of section headings that actually occur in the corpus:
"<Field>-<language>:" for Title/Description/Claims/Abstract, plus the bare
"Claims:"/"Abstract:" variant used by a handful of files.

A line is only treated as a section heading (see SectionDetector) if it
normalizes to one of these — this replaces the old "any short colon/
ALL-CAPS line" heuristic, which misfired on CJK claim/title text (CJK
characters have no case, so `line == line.upper()` is trivially true for
them) and shredded Title-chinese / Claims-chinese sections into dozens of
fake headings.
"""

from __future__ import annotations

_HEADING_FIELDS = ("title", "description", "claims", "abstract")
_HEADING_LANGUAGES = (
    "english",
    "french",
    "german",
    "chinese",
    "spanish",
    "japanese",
    "korean",
)

KNOWN_SECTION_HEADINGS: frozenset[str] = frozenset(
    {f"{field}-{lang}" for field in _HEADING_FIELDS for lang in _HEADING_LANGUAGES}
    | {"claims", "abstract"}
)

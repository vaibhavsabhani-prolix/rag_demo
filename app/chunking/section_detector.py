"""
Section Detector

Dynamically detects section headings in structured documents.
No hardcoded field names — uses heuristic patterns only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import (
    ALLCAPS_MIN_ALPHA,
    MAX_HEADING_LENGTH,
    MAX_HEADING_WORDS,
)

# ==================================================================
# Data model
# ==================================================================


@dataclass
class Section:
    """
    A detected section with its heading, body content, and character
    offsets within the original document text.
    """

    heading: str
    content: str

    # Character offsets relative to the full document text
    start_offset: int = 0
    end_offset: int = 0


# ==================================================================
# Heading patterns (generic, no domain-specific words)
# ==================================================================

# Numbered heading:  "1.", "1.1", "1.1.2", "1)", "1.1)"
_NUMBERED_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.)]\s+\S")

# Markdown-style heading:  "# Heading", "## Sub-heading"
_MARKDOWN_RE = re.compile(r"^\s*#{1,6}\s+\S")


# ==================================================================
# Detector
# ==================================================================


class SectionDetector:
    """
    Generic section detector for structured documents.

    Detection heuristics (none are domain-specific):

    1. Line ends with ':'  and is short   → heading
    2. Numbered prefix (1., 1.1, 1))      → heading
    3. Markdown-style (# / ## / ###)      → heading
    4. ALL-CAPS short line                 → heading
    """

    def __init__(
        self,
        max_heading_length: int = MAX_HEADING_LENGTH,
        max_heading_words: int = MAX_HEADING_WORDS,
        allcaps_min_alpha: int = ALLCAPS_MIN_ALPHA,
    ) -> None:

        self.max_heading_length = max_heading_length
        self.max_heading_words = max_heading_words
        self.allcaps_min_alpha = allcaps_min_alpha

    # ==============================================================
    # Heading detection
    # ==============================================================

    def is_heading(self, line: str) -> bool:
        """
        Return True if *line* looks like a section heading.
        """

        stripped = line.strip()

        if not stripped:
            return False

        # Quick length gate — long lines are never headings
        if len(stripped) > self.max_heading_length:
            return False

        word_count = len(stripped.split())

        if word_count > self.max_heading_words:
            return False

        # ---- Pattern 1: colon-terminated short line ----
        if stripped.endswith(":") and word_count <= self.max_heading_words:
            return True

        # ---- Pattern 2: numbered heading ----
        if _NUMBERED_RE.match(stripped) and stripped.endswith(":"):
            return True

        # ---- Pattern 3: markdown heading ----
        if _MARKDOWN_RE.match(stripped):
            return True

        # ---- Pattern 4: ALL-CAPS line ----
        alpha_chars = sum(1 for c in stripped if c.isalpha())

        if (
            alpha_chars >= self.allcaps_min_alpha
            and stripped == stripped.upper()
            and word_count <= self.max_heading_words
        ):
            return True

        return False

    # ==============================================================
    # Clean heading text
    # ==============================================================

    @staticmethod
    def clean_heading(line: str) -> str:
        """
        Normalise a heading line for use as a section label.

        Removes trailing colons, leading '#' markers,
        and excess whitespace.
        """

        cleaned = line.strip()

        # Remove markdown markers
        cleaned = re.sub(r"^#+\s*", "", cleaned)

        # Remove trailing colon
        cleaned = cleaned.rstrip(":")

        return cleaned.strip()

    # ==============================================================
    # Split document into sections
    # ==============================================================

    def detect(self, text: str) -> list[Section]:
        """
        Split *text* into sections.

        Everything from one heading to the next belongs to that heading.
        Content before the first detected heading goes into a section
        with heading "Document".

        Each Section carries character offsets (start_offset, end_offset)
        relative to the original *text*, enabling downstream components
        to track chunk positions within the full document.
        """

        lines = text.splitlines(keepends=True)

        sections: list[Section] = []

        current_heading = "Document"
        current_lines: list[str] = []
        # Character offset where the current section's content begins
        section_content_start: int = 0
        # Running character position
        char_pos: int = 0
        # Whether we've just started a new heading (to capture content start)
        heading_just_set: bool = True

        for line in lines:
            line_len = len(line)

            stripped = line.rstrip("\n").rstrip("\r")

            if self.is_heading(stripped):

                # Flush accumulated content
                if current_lines:
                    content = "".join(current_lines).strip()
                    if content:
                        # start_offset = where this section's content starts
                        # end_offset = where it ends (exclusive)
                        content_start = section_content_start
                        content_end = char_pos
                        sections.append(
                            Section(
                                heading=current_heading,
                                content=content,
                                start_offset=content_start,
                                end_offset=content_end,
                            )
                        )

                current_heading = self.clean_heading(stripped)
                current_lines = []
                heading_just_set = True

            else:
                if heading_just_set:
                    section_content_start = char_pos
                    heading_just_set = False
                current_lines.append(line)

            char_pos += line_len

        # Flush final section
        if current_lines:
            content = "".join(current_lines).strip()

            if content:
                sections.append(
                    Section(
                        heading=current_heading,
                        content=content,
                        start_offset=section_content_start,
                        end_offset=char_pos,
                    )
                )

        return sections

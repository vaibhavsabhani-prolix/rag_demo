"""
Query Match Highlighting

Finds the parts of a chunk's text that match the user's query, so the UI can
show *why* a chunk was retrieved:

- Sentence spans: Phase 6 sends each chunk's sentences to the BGE
  cross-encoder along with the chunks themselves, in the same batched
  requests, and sentences scoring above a threshold are marked. This catches
  matches by meaning ("condenses atmospheric moisture" for "generates water
  from air"), in any language the reranker covers.
- Term spans: occurrences of the query's own words (and the parsed query's
  concepts), with light stemming so "produces" also marks "produced" and
  "production".

Spans are character offsets into the original text.
"""

import re
from typing import List, Optional, Sequence, Tuple

from app.config import (
    HIGHLIGHT_MIN_SENTENCE_CHARS,
    HIGHLIGHT_MIN_SENTENCE_WORDS,
    HIGHLIGHT_SENTENCE_STRONG_THRESHOLD,
    HIGHLIGHT_SENTENCE_THRESHOLD,
)
from app.models.reranking import ChunkHighlight, HighlightSentence, HighlightTerm

Span = Tuple[int, int]

# A sentence ends after . ! ? (plus closing quotes/brackets) followed by
# whitespace, after CJK sentence punctuation or a semicolon (patent claims list
# their elements with semicolons), or at a line break.
_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*\s+|[。！？；;]+\s*|\n\s*")

# Words whose trailing period does not end a sentence ("see FIG. 2", "e.g. a").
_ABBREVIATIONS = {
    "fig", "figs", "e.g", "i.e", "etc", "no", "nos", "approx", "al", "vs",
    "ref", "ser", "pat", "u.s", "cf", "resp", "ca", "eq", "sec", "vol",
}

# Han ideographs and katakana. Hiragana is left out: in Japanese it is mostly
# grammatical particles and verb endings, so it separates the content words.
_CJK_RUN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿゠-ヿ]+")
_CJK_CHAR_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_HANGUL_RE = re.compile(r"[가-힯]")
_WORD_RE = re.compile(r"\w+")

_STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "does", "for", "from", "has", "have", "how", "in", "into", "is",
    "it", "its", "may", "not", "of", "on", "or", "our", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "this", "those", "through",
    "to", "using", "was", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "within", "without", "would", "any", "all", "some",
    "find", "patent", "patents", "invention", "method", "methods", "system",
    "systems", "device", "devices", "apparatus", "related", "about", "based",
    "used", "use", "uses", "wherein", "thereof", "comprising", "including",
    # French / Spanish / German function words
    "les", "des", "une", "pour", "avec", "dans", "par", "sur", "qui", "est",
    "los", "las", "con", "para", "por", "del", "que", "una", "der", "die",
    "das", "und", "mit", "für", "von", "den", "dem", "ein", "eine", "zur",
}

# Longest first, so "ations" is tried before "s".
_SUFFIXES = ("ations", "ation", "ions", "ion", "ings", "ing", "ers", "er",
             "edly", "ed", "es", "ly", "s")
_MIN_STEM = 4


def split_sentences(text: str) -> List[Span]:
    """Split *text* into sentence spans, trimmed of surrounding whitespace."""
    spans: List[Span] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.group(0)[0] == "." and _ends_with_abbreviation(text[start : m.start()]):
            continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(text)))

    trimmed: List[Span] = []
    for s, e in spans:
        segment = text[s:e]
        s += len(segment) - len(segment.lstrip())
        e -= len(segment) - len(segment.rstrip())
        if e > s:
            trimmed.append((s, e))
    return trimmed


def _ends_with_abbreviation(before: str) -> bool:
    words = before.split()
    if not words:
        return False
    last = words[-1].lower().rstrip(".")
    # Single letters are initials or list markers ("J. Smith", "a. the valve").
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            word = word[: -len(suffix)]
            break
    # "generate" -> "generat" so it also covers "generating" / "generation".
    if word.endswith("e") and len(word) - 1 >= _MIN_STEM:
        word = word[:-1]
    return word


def extract_term_patterns(query: str, extra_terms: Sequence[str] = ()) -> Optional[re.Pattern]:
    """
    Build one case-insensitive regex matching the content words of the query
    and of *extra_terms* (e.g. the parsed query's concepts).
    """
    word_stems: set[str] = set()
    cjk_literals: set[str] = set()

    for source in (query, *extra_terms):
        for run in _CJK_RUN_RE.findall(source):
            # Chinese and Japanese have no word spaces, so match the run itself
            # and every two-character window of it (most CJK words are 2 chars).
            if len(run) >= 2:
                cjk_literals.add(run)
                cjk_literals.update(run[i : i + 2] for i in range(len(run) - 1))
        for word in _WORD_RE.findall(source.lower()):
            if _CJK_CHAR_RE.search(word) or word.isdigit() or word in _STOPWORDS:
                continue
            min_len = 2 if _HANGUL_RE.search(word) else 3
            if len(word) < min_len:
                continue
            word_stems.add(_stem(word) if word.isascii() else word)

    alternatives: List[str] = []
    # Latin/Hangul words: match at a word start, allowing any ending.
    alternatives += [rf"\b{re.escape(stem)}\w*" for stem in word_stems]
    alternatives += [re.escape(lit) for lit in cjk_literals]
    if not alternatives:
        return None
    # Longest first so a whole CJK run wins over its bigrams.
    alternatives.sort(key=len, reverse=True)
    return re.compile("|".join(alternatives), re.IGNORECASE)


def find_term_spans(text: str, pattern: Optional[re.Pattern]) -> List[Span]:
    """Non-overlapping spans of query terms in *text*, adjacent ones merged."""
    if pattern is None:
        return []
    spans: List[Span] = []
    for m in pattern.finditer(text):
        if m.end() == m.start():
            continue
        if spans and m.start() <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], m.end()))
        else:
            spans.append((m.start(), m.end()))
    return spans


def sentences_to_score(
    text: str,
    min_chars: int = HIGHLIGHT_MIN_SENTENCE_CHARS,
    min_words: int = HIGHLIGHT_MIN_SENTENCE_WORDS,
) -> List[Span]:
    """
    Sentence spans of *text* worth scoring. Fragments like "FIG. 1" and short
    headings are skipped; the word minimum does not apply to Chinese/Japanese,
    which have no spaces between words.
    """
    return [
        (s, e) for s, e in split_sentences(text)
        if e - s >= min_chars
        and (_CJK_CHAR_RE.search(text[s:e]) or len(text[s:e].split()) >= min_words)
    ]


def build_chunk_highlight(
    text: str,
    sentence_spans: Sequence[Span],
    sentence_scores: Sequence[float],
    term_pattern: Optional[re.Pattern],
    threshold: float = HIGHLIGHT_SENTENCE_THRESHOLD,
    strong_threshold: float = HIGHLIGHT_SENTENCE_STRONG_THRESHOLD,
) -> ChunkHighlight:
    """Combine sentence scores and query-term matches into a chunk's highlight."""
    return ChunkHighlight(
        sentences=[
            HighlightSentence(start=s, end=e, score=round(score, 4), strong=score >= strong_threshold)
            for (s, e), score in zip(sentence_spans, sentence_scores)
            if score >= threshold
        ],
        terms=[HighlightTerm(start=s, end=e) for s, e in find_term_spans(text, term_pattern)],
    )

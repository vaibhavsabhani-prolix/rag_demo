"""
Evidence Selector & Answer Extractor

Identifies answer-bearing evidence for question-type queries and extracts
the verbatim answer phrase, supporting evidence sentence(s), and exact character spans
for visual highlighting in the search results.

Strict rules:
1. Dynamic across all domains - no hardcoded sensor, material, or process rules.
2. The extracted answer and evidence MUST come directly from the retrieved patent text.
3. Character offsets MUST reference the original chunk text directly: chunk.text[start:end].
4. No hallucination - non-answering chunks receive no answer extraction.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from app.config import (
    QUERY_LLM_REMOTE_API_KEY,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
)
from app.models.patent_search_result import AnswerEvidence
from app.query_understanding.models import ParsedQuery

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "been",
    "can",
    "has",
    "have",
    "had",
    "will",
    "would",
    "could",
    "should",
    "what",
    "which",
    "when",
    "where",
    "how",
    "why",
    "who",
    "whom",
    "whose",
    "does",
    "done",
    "into",
    "onto",
    "upon",
    "also",
    "such",
    "than",
    "then",
    "them",
    "they",
    "their",
    "there",
    "based",
    "other",
    "about",
    "used",
    "includes",
    "including",
    "included",
    "performed",
    "determines",
    "determined",
    "type",
    "types",
    "kind",
    "kinds",
}


def _tokenize_terms(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z0-9][a-z0-9\-]*\b", (text or "").lower())
    normalized: list[str] = []
    for token in tokens:
        term = _normalize_token(token)
        if len(term) > 2 and term not in _STOP_WORDS:
            normalized.append(term)
    return normalized


def _normalize_token(token: str) -> str:
    t = (token or "").strip().lower()
    if not t:
        return ""
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("tion") and len(t) > 6:
        t = t[:-3]
    for suffix in ("ing", "ed", "es", "s"):
        if t.endswith(suffix) and len(t) > len(suffix) + 2:
            t = t[: -len(suffix)]
            break
    return t


def _extract_phrases(text: str) -> list[str]:
    terms = _tokenize_terms(text)
    phrases: list[str] = []
    for n in (3, 2):
        for i in range(max(0, len(terms) - n + 1)):
            phrase = " ".join(terms[i : i + n])
            if phrase:
                phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def _coverage_score(text_lower: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    normalized_text_terms = set(_tokenize_terms(text_lower))
    normalized_text = " ".join(_tokenize_terms(text_lower))
    hits = 0
    for term in terms:
        if " " in term:
            if term in normalized_text:
                hits += 1
        elif term in normalized_text_terms:
            hits += 1
    return hits / len(terms)


def _match_terms(text_lower: str, terms: list[str]) -> list[str]:
    normalized_text_terms = set(_tokenize_terms(text_lower))
    normalized_text = " ".join(_tokenize_terms(text_lower))
    matches: list[str] = []
    for term in terms:
        if " " in term:
            if term in normalized_text:
                matches.append(term)
        elif term in normalized_text_terms:
            matches.append(term)
    return matches


def locate_span_in_text(
    chunk_text: str, evidence_snippet: str
) -> tuple[int, int] | None:
    """
    Robustly locate the character span [start_char, end_char] of evidence_snippet
    inside chunk_text.

    Tolerates:
    - Leading/trailing quotation marks or whitespace in evidence_snippet
    - Minor whitespace differences (e.g. multiple spaces vs newlines)
    - Case differences

    Always returns exact slice indices into the ORIGINAL chunk_text such that
    chunk_text[start_char:end_char] contains the matched text.
    """
    if not chunk_text or not evidence_snippet:
        return None

    snippet = evidence_snippet.strip(" \t\n\r\"'“”‘’`")
    if not snippet or len(snippet) < 3:
        return None

    # 1. Exact match
    idx = chunk_text.find(snippet)
    if idx != -1:
        return (idx, idx + len(snippet))

    # 2. Case-insensitive exact match
    idx = chunk_text.lower().find(snippet.lower())
    if idx != -1:
        return (idx, idx + len(snippet))

    # 3. Whitespace-flexible regex match
    tokens = re.split(r"\s+", snippet)
    if len(tokens) > 1:
        escaped_tokens = [re.escape(t) for t in tokens if t]
        pattern = r"\s+".join(escaped_tokens)
        match = re.search(pattern, chunk_text, re.IGNORECASE)
        if match:
            return (match.start(), match.end())

    # 4. Long snippet prefix+suffix boundary match
    if len(snippet) > 40:
        prefix = snippet[:25]
        suffix = snippet[-25:]
        p_match = re.search(re.escape(prefix), chunk_text, re.IGNORECASE)
        s_match = re.search(re.escape(suffix), chunk_text, re.IGNORECASE)
        if p_match and s_match and p_match.start() < s_match.end():
            span_len = s_match.end() - p_match.start()
            if abs(span_len - len(snippet)) < len(snippet) * 0.4:
                return (p_match.start(), s_match.end())

    return None


def _merge_spans(
    chunk_text: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Sort and merge overlapping or adjacent character spans."""
    if not chunk_text or not spans:
        return []
    text_len = len(chunk_text)
    valid_spans = [(s, e) for s, e in spans if 0 <= s < e <= text_len]
    if not valid_spans:
        return []
    valid_spans.sort(key=lambda x: (x[0], -x[1]))
    merged: list[tuple[int, int]] = []
    for current in valid_spans:
        if not merged:
            merged.append(current)
        else:
            prev_s, prev_e = merged[-1]
            if current[0] <= prev_e:
                merged[-1] = (prev_s, max(prev_e, current[1]))
            else:
                merged.append(current)
    return merged


def highlight_spans(chunk_text: str, spans: list[tuple[int, int]]) -> str:
    """
    Wrap the spans in <mark> HTML tags for UI rendering.
    Overlapping or adjacent spans are merged.
    Non-highlighted portions are HTML-escaped.
    """
    if not chunk_text:
        return ""
    merged = _merge_spans(chunk_text, spans)
    if not merged:
        return html.escape(chunk_text)

    parts: list[str] = []
    last_idx = 0
    mark_style = (
        "background-color: #ffe066; color: #111; padding: 2px 4px; "
        "border-radius: 3px; font-weight: 600;"
    )

    for start, end in merged:
        if start > last_idx:
            parts.append(html.escape(chunk_text[last_idx:start]))
        parts.append(
            f'<mark style="{mark_style}">{html.escape(chunk_text[start:end])}</mark>'
        )
        last_idx = end

    if last_idx < len(chunk_text):
        parts.append(html.escape(chunk_text[last_idx:]))

    return "".join(parts)


def highlight_span(text: str, span: tuple[int, int] | None) -> str:
    """Single-span convenience helper."""
    if span is None:
        return html.escape(text)
    return highlight_spans(text, [span])


class EvidenceSelector:
    """
    Dynamically extracts concise answers, verbatim supporting evidence,
    and exact character offsets for visual highlighting in patent search results.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._remote_client = None
        self._remote_available = False

        if self.use_llm:
            self._check_remote_llm()

    def _check_remote_llm(self):
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=QUERY_LLM_REMOTE_BASE_URL,
                api_key=QUERY_LLM_REMOTE_API_KEY,
                timeout=60.0,
            )
            client.models.list()
            self._remote_client = client
            self._remote_available = True
        except Exception:
            self._remote_client = None
            self._remote_available = False

    def select_and_extract(
        self,
        results: list[tuple[float, object]],
        parsed_query: ParsedQuery,
        max_chunks_to_extract: int = 15,
    ) -> list[tuple[float, object]]:
        """
        Process reranked candidate chunks for a query.
        - For question queries (is_question=True): Extracts answers, evidence sentences, and answer spans.
        - For normal queries (is_question=False): Dynamically highlights query concepts and requirements present in chunk.
        """
        if not results:
            return results

        # -------------------------------------------------------------
        # Path A: Normal Query Dynamic Highlighting (is_question == False)
        # -------------------------------------------------------------
        if not parsed_query.is_question:
            processed: list[tuple[float, object]] = []
            for idx, (score, chunk_point) in enumerate(results):
                payload = getattr(chunk_point, "payload", {}) or {}
                text = payload.get("text", "")
                if idx < max_chunks_to_extract and text:
                    extraction = self.highlight_normal_query(text, parsed_query)
                else:
                    extraction = {
                        "highlighted_text": html.escape(text),
                        "spans": [],
                        "matched_phrases": [],
                    }

                if hasattr(chunk_point, "payload") and chunk_point.payload is not None:
                    chunk_point.payload["highlighted_text"] = extraction[
                        "highlighted_text"
                    ]
                    chunk_point.payload["highlight_spans"] = extraction["spans"]
                    chunk_point.payload["answer"] = None
                    chunk_point.payload["answer_evidence"] = []
                    chunk_point.payload["answer_span"] = None
                    chunk_point.payload["answer_score"] = None

                processed.append((score, chunk_point))
            return processed

        # -------------------------------------------------------------
        # Path B: Question Query Answer & Evidence Extraction (is_question == True)
        # -------------------------------------------------------------
        processed: list[tuple[float, object]] = []

        for idx, (score, chunk_point) in enumerate(results):
            payload = getattr(chunk_point, "payload", {}) or {}
            text = payload.get("text", "")
            if idx < max_chunks_to_extract and text:
                extraction = self.extract_answer(text, parsed_query)
            else:
                extraction = {
                    "has_answer": False,
                    "answer": None,
                    "evidence_items": [],
                    "primary_evidence": None,
                    "primary_span": None,
                    "answer_score": 0.0,
                    "highlighted_text": html.escape(text),
                    "debug": {
                        "question": parsed_query.original_query,
                        "dynamic_answer_target": (
                            parsed_query.question_intent.target
                            if parsed_query.question_intent
                            else parsed_query.original_query
                        ),
                        "expected_answer_type": (
                            parsed_query.question_intent.expected_answer_type
                            if parsed_query.question_intent
                            else ""
                        ),
                        "answer_criteria": (
                            parsed_query.question_intent.answer_criteria
                            if parsed_query.question_intent
                            else ""
                        ),
                        "candidate_evidence": [],
                        "selected_evidence": None,
                        "extracted_answer": None,
                        "highlighted_span": None,
                        "reason": "Chunk skipped due to max_chunks_to_extract limit",
                    },
                }

            # Attach extraction to payload for downstream aggregation
            if hasattr(chunk_point, "payload") and chunk_point.payload is not None:
                chunk_point.payload["answer"] = extraction["answer"]
                chunk_point.payload["answer_evidence"] = extraction["evidence_items"]
                chunk_point.payload["answer_span"] = extraction["primary_span"]
                chunk_point.payload["answer_score"] = extraction["answer_score"]
                chunk_point.payload["highlighted_text"] = extraction["highlighted_text"]
                chunk_point.payload["answer_debug"] = extraction.get("debug")

            processed.append((score, chunk_point))

        return processed

    def highlight_normal_query(
        self, chunk_text: str, parsed_query: ParsedQuery
    ) -> dict[str, Any]:
        """
        Dynamically determine which query concepts and representations from ParsedQuery
        actually occur in the patent chunk, and highlight those exact character spans.
        """
        if not chunk_text or not chunk_text.strip():
            return {
                "highlighted_text": "",
                "spans": [],
                "matched_phrases": [],
            }

        candidates = self._collect_normal_highlight_candidates(parsed_query)
        if not candidates:
            return {
                "highlighted_text": html.escape(chunk_text),
                "spans": [],
                "matched_phrases": [],
            }

        all_spans: list[tuple[int, int]] = []
        matched_phrases: list[str] = []

        for cand in candidates:
            tokens = [t for t in re.split(r"\s+", cand) if t]
            if not tokens:
                continue
            escaped_tokens = [re.escape(t) for t in tokens if t]
            pattern = r"\b" + r"\s+".join(escaped_tokens) + r"\b"
            try:
                for match in re.finditer(pattern, chunk_text, re.IGNORECASE):
                    all_spans.append((match.start(), match.end()))
                    matched_phrases.append(chunk_text[match.start() : match.end()])
            except Exception:
                pass

        merged_spans = _merge_spans(chunk_text, all_spans)
        if not merged_spans:
            return {
                "highlighted_text": html.escape(chunk_text),
                "spans": [],
                "matched_phrases": [],
            }

        highlighted = highlight_spans(chunk_text, merged_spans)
        matched = [chunk_text[s:e] for s, e in merged_spans]
        return {
            "highlighted_text": highlighted,
            "spans": merged_spans,
            "matched_phrases": list(dict.fromkeys(matched)),
        }

    def _collect_normal_highlight_candidates(
        self, parsed_query: ParsedQuery
    ) -> list[str]:
        """
        Dynamically extracts candidate highlight phrases from ParsedQuery
        (concepts, semantic_variants, goals, constraints, relationships,
        requirements, optimization, semantic_query, original_query), prioritizing
        required/high-importance and multi-word items.
        """
        candidates: list[tuple[str, float, int]] = []
        seen = set()

        def add_candidate(text: str, score: float):
            if not text or not isinstance(text, str):
                return
            cleaned = text.strip(' \t\n\r"`.,:;()')
            if not cleaned or len(cleaned) < 2:
                return
            tokens = [
                t for t in re.split(r"\s+", cleaned) if t.lower() not in _STOP_WORDS
            ]
            if not tokens:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            word_count = len(tokens)
            weight = score * (1.0 + 0.4 * min(word_count - 1, 4))
            candidates.append((cleaned, weight, word_count))

            # Add singular/plural inflection
            if cleaned.endswith("s") and len(cleaned) > 4:
                sing = cleaned[:-1]
                if sing.lower() not in seen:
                    seen.add(sing.lower())
                    candidates.append((sing, score * 0.95, word_count))
            elif not cleaned.endswith("s") and len(cleaned) > 3:
                plur = cleaned + "s"
                if plur.lower() not in seen:
                    seen.add(plur.lower())
                    candidates.append((plur, score * 0.95, word_count))

        # 1. Concepts & variants
        for c in parsed_query.concepts:
            imp = c.importance if c.importance is not None else 0.8
            req_mult = 1.3 if c.required else 1.0
            add_candidate(c.text, imp * req_mult)
            for v in c.semantic_variants:
                add_candidate(v, imp * req_mult * 0.9)
            # Add individual non-stopword tokens from multi-word concepts with lower score
            c_tokens = [
                t for t in re.split(r"\s+", c.text) if t.lower() not in _STOP_WORDS
            ]
            for t in c_tokens:
                if len(t) > 2:
                    add_candidate(t, imp * req_mult * 0.7)

        # 2. Goals & keywords
        for g in parsed_query.goals:
            imp = g.importance if g.importance is not None else 0.7
            req_mult = 1.3 if g.required else 1.0
            add_candidate(g.text, imp * req_mult)
            for kw in g.keywords:
                add_candidate(kw, imp * req_mult * 0.9)

        # 3. Constraints & keywords
        for k in parsed_query.constraints:
            imp = k.importance if k.importance is not None else 0.7
            req_mult = 1.3 if k.required else 1.0
            add_candidate(k.text, imp * req_mult)
            for kw in k.keywords:
                add_candidate(kw, imp * req_mult * 0.9)

        # 4. Relationships
        for r in parsed_query.relationships:
            imp = r.importance if r.importance is not None else 0.7
            add_candidate(r.source, imp)
            add_candidate(r.target, imp)
            add_candidate(f"{r.source} {r.target}", imp * 1.1)

        # 5. Requirements
        for req in parsed_query.requirements:
            imp = req.importance if req.importance is not None else 0.7
            for kw in req.keywords:
                add_candidate(kw, imp * 0.9)
            if req.description:
                add_candidate(req.description, imp * 0.8)

        # 6. Optimization
        for opt in parsed_query.optimization:
            imp = opt.importance if opt.importance is not None else 0.7
            add_candidate(opt.property, imp)

        # 7. Semantic Query & Original Query
        for q_text, base_score in [
            (parsed_query.semantic_query, 0.75),
            (parsed_query.original_query, 0.70),
        ]:
            if q_text:
                add_candidate(q_text, base_score)
                q_tokens = [
                    t for t in re.split(r"\s+", q_text) if t.lower() not in _STOP_WORDS
                ]
                if len(q_tokens) >= 2:
                    for n in (3, 2):
                        for i in range(max(0, len(q_tokens) - n + 1)):
                            phrase = " ".join(q_tokens[i : i + n])
                            add_candidate(phrase, base_score * 0.85)
                for t in q_tokens:
                    if len(t) > 2:
                        add_candidate(t, base_score * 0.65)

        # Sort by weight descending, then by word count descending, then by string length
        candidates.sort(key=lambda x: (x[1], x[2], len(x[0])), reverse=True)
        return [c[0] for c in candidates]

    def extract_answer(self, chunk_text: str, parsed_query: ParsedQuery) -> dict:
        """
        Extract concise answer, supporting evidence, and character spans from chunk_text.
        """
        if not chunk_text.strip():
            return {
                "has_answer": False,
                "answer": None,
                "evidence_items": [],
                "primary_evidence": None,
                "primary_span": None,
                "answer_score": 0.0,
                "highlighted_text": "",
            }

        # Try LLM extraction if available
        if self.use_llm and self._remote_available:
            extracted = self._extract_with_remote_llm(chunk_text, parsed_query)
            if extracted is not None:
                return extracted

        # Deterministic extraction fallback
        return self._extract_deterministic(chunk_text, parsed_query)

    def _extract_with_remote_llm(
        self, chunk_text: str, parsed_query: ParsedQuery
    ) -> dict | None:
        try:
            target = ""
            criteria = ""
            expected_type = ""
            if parsed_query.question_intent:
                target = parsed_query.question_intent.target
                criteria = parsed_query.question_intent.answer_criteria
                expected_type = parsed_query.question_intent.expected_answer_type

            prompt = f"""<|im_start|>system
You are a precision Answer and Evidence Extractor for a patent search engine.
Analyze the provided patent passage to determine if and how it answers the user's question.

STRICT INSTRUCTIONS:
1. "answer": Provide a concise, clear summary/extraction of the answer to the question.
2. "evidence_spans": Provide 1 or more verbatim sentences/phrases directly from the PASSAGE that support the answer.
3. Every item in "evidence_spans" MUST be copied verbatim from the PASSAGE so it can be located by character offsets.
4. If the passage does NOT answer the question, set "has_answer": false, "answer": null, "evidence_spans": [].
5. Return STRICT JSON only.

JSON Format:
{{
  "has_answer": true | false,
  "answer": "concise answer to question" | null,
  "evidence_spans": [
    {{"evidence_text": "verbatim sentence from passage", "confidence": 0.0 to 1.0}}
  ]
}}
<|im_end|>
<|im_start|>user
QUESTION: {parsed_query.original_query}
TARGET: {target} (type: {expected_type})
CRITERIA: {criteria}

PATENT PASSAGE:
\"\"\"{chunk_text}\"\"\"
<|im_end|>
<|im_start|>assistant
"""
            response = self._remote_client.chat.completions.create(
                model=QUERY_LLM_REMOTE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            cleaned = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None

            data = json.loads(match.group(0))
            has_answer = bool(data.get("has_answer"))
            raw_answer = data.get("answer")
            raw_evidence_spans = data.get("evidence_spans", [])

            if not has_answer or not raw_answer:
                return {
                    "has_answer": False,
                    "answer": None,
                    "evidence_items": [],
                    "primary_evidence": None,
                    "primary_span": None,
                    "answer_score": 0.0,
                    "highlighted_text": html.escape(chunk_text),
                    "debug": {
                        "question": parsed_query.original_query,
                        "dynamic_answer_target": target,
                        "expected_answer_type": expected_type,
                        "answer_criteria": criteria,
                        "candidate_evidence": [],
                        "selected_evidence": None,
                        "extracted_answer": None,
                        "highlighted_span": None,
                        "reason": "Remote extractor reported no direct answer",
                    },
                }

            answer_str = str(raw_answer).strip()
            evidence_items: list[AnswerEvidence] = []
            spans: list[tuple[int, int]] = []

            for item in raw_evidence_spans:
                if isinstance(item, dict):
                    ev_text = str(item.get("evidence_text", "")).strip()
                    conf = float(item.get("confidence", 0.9) or 0.9)
                elif isinstance(item, str):
                    ev_text = item.strip()
                    conf = 0.9
                else:
                    continue

                span = locate_span_in_text(chunk_text, ev_text)
                if span:
                    matched_text = chunk_text[span[0] : span[1]]
                    evidence_items.append(
                        AnswerEvidence(
                            answer=answer_str,
                            evidence_text=matched_text,
                            start_char=span[0],
                            end_char=span[1],
                            confidence=conf,
                        )
                    )
                    spans.append(span)

            # If evidence_spans couldn't be located, try finding the answer itself
            if not spans:
                span = locate_span_in_text(chunk_text, answer_str)
                if span:
                    matched_text = chunk_text[span[0] : span[1]]
                    evidence_items.append(
                        AnswerEvidence(
                            answer=answer_str,
                            evidence_text=matched_text,
                            start_char=span[0],
                            end_char=span[1],
                            confidence=0.9,
                        )
                    )
                    spans.append(span)

            if not spans:
                # Cannot locate verified span in original chunk text
                return self._extract_deterministic(chunk_text, parsed_query)

            primary_span = spans[0]
            primary_evidence = evidence_items[0].evidence_text
            highlighted = highlight_spans(chunk_text, spans)

            return {
                "has_answer": True,
                "answer": answer_str,
                "evidence_items": evidence_items,
                "primary_evidence": primary_evidence,
                "primary_span": primary_span,
                "answer_score": max(ev.confidence or 0.0 for ev in evidence_items),
                "highlighted_text": highlighted,
                "debug": {
                    "question": parsed_query.original_query,
                    "dynamic_answer_target": target,
                    "expected_answer_type": expected_type,
                    "answer_criteria": criteria,
                    "candidate_evidence": [
                        {
                            "evidence_text": ev.evidence_text,
                            "confidence": ev.confidence,
                            "span": [ev.start_char, ev.end_char],
                        }
                        for ev in evidence_items
                    ],
                    "selected_evidence": {
                        "text": primary_evidence,
                        "span": [primary_span[0], primary_span[1]],
                    },
                    "extracted_answer": answer_str,
                    "highlighted_span": [primary_span[0], primary_span[1]],
                    "reason": "Remote extraction selected direct answer evidence",
                },
            }
        except Exception:
            return None

    def _extract_deterministic(
        self, chunk_text: str, parsed_query: ParsedQuery
    ) -> dict:
        """
        Deterministic sentence-level answer and evidence extraction.
        Finds the sentence most relevant to the question intent and locates exact spans.
        """
        sentences = self._split_sentences(chunk_text)
        if not sentences:
            return {
                "has_answer": False,
                "answer": None,
                "evidence_items": [],
                "primary_evidence": None,
                "primary_span": None,
                "answer_score": 0.0,
                "highlighted_text": html.escape(chunk_text),
            }

        profile = self._build_answer_target_profile(parsed_query)
        if not profile["query_terms"]:
            return {
                "has_answer": False,
                "answer": None,
                "evidence_items": [],
                "primary_evidence": None,
                "primary_span": None,
                "answer_score": 0.0,
                "highlighted_text": html.escape(chunk_text),
                "debug": {
                    "question": parsed_query.original_query,
                    "dynamic_answer_target": profile["target_text"],
                    "expected_answer_type": profile["expected_answer_type"],
                    "answer_criteria": profile["answer_criteria"],
                    "candidate_evidence": [],
                    "selected_evidence": None,
                    "extracted_answer": None,
                    "highlighted_span": None,
                    "reason": "No dynamic target terms available",
                },
            }

        scored_sentences: list[dict[str, Any]] = []

        for sent_text, sent_start, sent_end in sentences:
            sent_lower = sent_text.lower()
            core_matches = _match_terms(sent_lower, profile["core_terms"])
            anchor_matches = _match_terms(sent_lower, profile["anchor_terms"])
            context_matches = _match_terms(sent_lower, profile["context_terms"])
            query_matches = _match_terms(sent_lower, profile["query_terms"])
            core_phrase_matches = _match_terms(sent_lower, profile["core_phrases"])
            anchor_phrase_matches = _match_terms(sent_lower, profile["anchor_phrases"])

            core_score = _coverage_score(sent_lower, profile["core_terms"])
            anchor_score = _coverage_score(sent_lower, profile["anchor_terms"])
            context_score = _coverage_score(sent_lower, profile["context_terms"])
            query_score = _coverage_score(sent_lower, profile["query_terms"])
            phrase_score = _coverage_score(sent_lower, profile["core_phrases"])
            anchor_phrase_score = _coverage_score(sent_lower, profile["anchor_phrases"])

            directness = (
                0.45 * anchor_score
                + 0.1 * anchor_phrase_score
                + 0.25 * core_score
                + 0.1 * phrase_score
                + 0.1 * query_score
                + 0.05 * context_score
            )
            distractor_penalty = max(0.0, context_score - max(core_score, phrase_score))
            final_score = max(0.0, directness - 0.65 * distractor_penalty)

            is_direct = (
                final_score >= 0.24
                and (
                    not profile["anchor_terms"]
                    or anchor_score >= 0.2
                    or anchor_phrase_score > 0.0
                    or len(anchor_matches) >= 1
                )
                and (core_score >= 0.2 or phrase_score > 0.0 or len(core_matches) >= 2)
            )

            if final_score >= 0.16 or is_direct:
                scored_sentences.append(
                    {
                        "text": sent_text,
                        "start": sent_start,
                        "end": sent_end,
                        "final_score": round(final_score, 4),
                        "directness": round(directness, 4),
                        "distractor_penalty": round(distractor_penalty, 4),
                        "core_score": round(core_score, 4),
                        "anchor_score": round(anchor_score, 4),
                        "context_score": round(context_score, 4),
                        "query_score": round(query_score, 4),
                        "core_phrase_score": round(phrase_score, 4),
                        "anchor_phrase_score": round(anchor_phrase_score, 4),
                        "anchor_matches": anchor_matches,
                        "core_matches": core_matches,
                        "context_matches": context_matches,
                        "query_matches": query_matches,
                        "core_phrase_matches": core_phrase_matches,
                        "anchor_phrase_matches": anchor_phrase_matches,
                        "is_direct": is_direct,
                    }
                )

        scored_sentences.sort(
            key=lambda s: (
                1 if s["is_direct"] else 0,
                s["final_score"],
                s["core_score"],
            ),
            reverse=True,
        )

        if not scored_sentences:
            return {
                "has_answer": False,
                "answer": None,
                "evidence_items": [],
                "primary_evidence": None,
                "primary_span": None,
                "answer_score": 0.0,
                "highlighted_text": html.escape(chunk_text),
                "debug": {
                    "question": parsed_query.original_query,
                    "dynamic_answer_target": profile["target_text"],
                    "expected_answer_type": profile["expected_answer_type"],
                    "answer_criteria": profile["answer_criteria"],
                    "candidate_evidence": [],
                    "selected_evidence": None,
                    "extracted_answer": None,
                    "highlighted_span": None,
                    "reason": "No candidate sentence reached minimum relevance",
                },
            }

        direct_sentences = [s for s in scored_sentences if s["is_direct"]]
        if not direct_sentences:
            return {
                "has_answer": False,
                "answer": None,
                "evidence_items": [],
                "primary_evidence": None,
                "primary_span": None,
                "answer_score": 0.0,
                "highlighted_text": html.escape(chunk_text),
                "debug": {
                    "question": parsed_query.original_query,
                    "dynamic_answer_target": profile["target_text"],
                    "expected_answer_type": profile["expected_answer_type"],
                    "answer_criteria": profile["answer_criteria"],
                    "candidate_evidence": scored_sentences[:5],
                    "selected_evidence": None,
                    "extracted_answer": "(no direct answer identified)",
                    "highlighted_span": None,
                    "reason": "Only related/context evidence found; no direct answer evidence",
                },
            }

        # Select top 1-2 direct answer sentences.
        top_sentences = [direct_sentences[0]]
        if len(direct_sentences) > 1 and direct_sentences[1]["final_score"] >= 0.22:
            top_sentences.append(direct_sentences[1])

        evidence_items: list[AnswerEvidence] = []
        spans: list[tuple[int, int]] = []
        best_answer_phrase = None

        for sent in top_sentences:
            sent_text = sent["text"]
            sent_start = sent["start"]
            sent_end = sent["end"]
            spans.append((sent_start, sent_end))
            matched_evidence_text = chunk_text[sent_start:sent_end]

            # Pick the longest matched core phrase/term present in evidence as answer phrase.
            if best_answer_phrase is None:
                ranked_matches = sent["core_phrase_matches"] + sent["core_matches"]
                for m in sorted(ranked_matches, key=len, reverse=True):
                    idx = sent_text.lower().find(m)
                    if idx != -1:
                        best_answer_phrase = sent_text[idx : idx + len(m)]
                        break
                if best_answer_phrase is None:
                    best_answer_phrase = sent_text

            evidence_items.append(
                AnswerEvidence(
                    answer=best_answer_phrase,
                    evidence_text=matched_evidence_text,
                    start_char=sent_start,
                    end_char=sent_end,
                    confidence=round(min(1.0, 0.45 + sent["final_score"] * 0.55), 3),
                )
            )

        primary_span = spans[0]
        primary_evidence = evidence_items[0].evidence_text
        highlighted = highlight_spans(chunk_text, spans)

        selected_debug = {
            "text": primary_evidence,
            "span": [primary_span[0], primary_span[1]],
            "direct_answer_relevance": top_sentences[0]["final_score"],
            "core_matches": top_sentences[0]["core_matches"],
            "anchor_matches": top_sentences[0]["anchor_matches"],
            "core_phrase_matches": top_sentences[0]["core_phrase_matches"],
            "context_matches": top_sentences[0]["context_matches"],
        }

        return {
            "has_answer": True,
            "answer": best_answer_phrase or primary_evidence,
            "evidence_items": evidence_items,
            "primary_evidence": primary_evidence,
            "primary_span": primary_span,
            "answer_score": evidence_items[0].confidence or 0.8,
            "highlighted_text": highlighted,
            "debug": {
                "question": parsed_query.original_query,
                "dynamic_answer_target": profile["target_text"],
                "expected_answer_type": profile["expected_answer_type"],
                "answer_criteria": profile["answer_criteria"],
                "candidate_evidence": scored_sentences[:5],
                "selected_evidence": selected_debug,
                "extracted_answer": best_answer_phrase or primary_evidence,
                "highlighted_span": [primary_span[0], primary_span[1]],
                "reason": "Selected highest direct-answer relevance sentence",
            },
        }

    def _build_answer_target_profile(self, parsed_query: ParsedQuery) -> dict[str, Any]:
        intent = parsed_query.question_intent
        target_text = (intent.target if intent else "") or parsed_query.original_query
        expected_answer_type = intent.expected_answer_type if intent else ""
        answer_criteria = intent.answer_criteria if intent else ""

        core_source = " ".join(
            part
            for part in [target_text, expected_answer_type]
            if part and part.strip()
        )
        core_terms = _tokenize_terms(core_source)
        core_phrases = _extract_phrases(core_source)

        context_parts = [answer_criteria]
        context_parts.extend(c.text for c in parsed_query.concepts)
        context_parts.extend(
            v for c in parsed_query.concepts for v in c.semantic_variants
        )
        context_parts.extend(g.text for g in parsed_query.goals)
        context_parts.extend(k.text for k in parsed_query.constraints)
        context_parts.extend(r.source for r in parsed_query.relationships)
        context_parts.extend(r.target for r in parsed_query.relationships)
        context_source = " ".join(part for part in context_parts if part)
        context_terms = _tokenize_terms(context_source)

        query_terms = _tokenize_terms(parsed_query.original_query)

        if not core_terms:
            core_terms = query_terms[:]
            core_phrases = _extract_phrases(parsed_query.original_query)

        # Anchor terms focus on the primary requested entity/action phrase.
        anchor_source_parts = re.split(
            r"\b(?:for|of|in|with|using|by|from|based\s+on)\b",
            target_text,
            flags=re.IGNORECASE,
        )
        anchor_source = (
            anchor_source_parts[0] if anchor_source_parts else target_text
        ).strip()
        if len(_tokenize_terms(anchor_source)) < 2 and len(anchor_source_parts) > 1:
            anchor_source = " ".join(
                part.strip() for part in anchor_source_parts[:2] if part.strip()
            )
        anchor_terms = _tokenize_terms(anchor_source)
        if not anchor_terms:
            anchor_terms = core_terms[: min(3, len(core_terms))]
        anchor_phrases = _extract_phrases(anchor_source)

        # Keep deterministic uniqueness and remove trivial overlaps from context bucket.
        core_terms = list(dict.fromkeys(core_terms))
        core_term_set = set(core_terms)
        context_terms = [
            t for t in dict.fromkeys(context_terms) if t not in core_term_set
        ]
        query_terms = list(dict.fromkeys(query_terms))
        anchor_terms = list(dict.fromkeys(anchor_terms))
        anchor_phrases = [
            phrase
            for phrase in dict.fromkeys(anchor_phrases)
            if phrase and all(part in core_term_set for part in phrase.split())
        ]

        return {
            "target_text": target_text,
            "expected_answer_type": expected_answer_type,
            "answer_criteria": answer_criteria,
            "core_terms": core_terms,
            "core_phrases": core_phrases,
            "anchor_terms": anchor_terms,
            "anchor_phrases": anchor_phrases,
            "context_terms": context_terms,
            "query_terms": query_terms,
        }

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with start and end character offsets."""
        sentences = []
        for match in re.finditer(r"[^.!?\n]+[.!?]?", text):
            s = match.group(0).strip()
            if s and len(s) > 5:
                start = match.start()
                end = match.end()
                sentences.append((s, start, end))
        return sentences

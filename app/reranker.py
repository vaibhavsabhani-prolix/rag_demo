"""
Reranker

Scores every candidate CHUNK individually against the user's query,
using a cross-encoder relevance model served by a remote reranking
server.

A candidate PATENT's score is the MAX of its own chunks' individual
scores. `results` is expected to already be a patent's COMPLETE indexed
chunk set (see SemanticSearch.search_detailed, which fetches every
chunk of every surviving candidate patent via
QdrantDB.get_chunks_for_patent_ids before calling rerank() - not just
whichever chunks the initial vector search happened to surface), so
"checking all chunks" is a retrieval-level property, not something this
module can enforce on its own. The chunk that achieved a patent's max
score is naturally its best-evidence chunk: its own relevance score is
stashed on `chunk.payload["_chunk_relevance"]` so SemanticSearch can
pick which chunk to display using the SAME model that decided
relevance, instead of a separate, weaker heuristic.

There is no hand-tuned weighting: the model's own 0-1 relevance
judgment, scaled to 0-10, IS the score. The only non-model-derived step
is a deterministic exclusion hard-filter (see _chunk_mentions()) for
literal "without X" style negation, which encoder relevance scores are
known to handle poorly on their own - term overlap with the excluded
term can outscore genuine relevance. That filter checks ONLY the single
chunk that actually earned a patent its score, never the patent's other
chunks: Query Understanding's exclusion extraction is itself an LLM
call and can over-infer terms the user never asked to exclude (e.g.
inferring "not video" from a query about "still images"), and a long
patent will very often mention an ordinary, unrelated technical term
like that somewhere in an irrelevant chunk. Checking every chunk a
patent has for a hallucinated exclusion would wrongly sink an otherwise
perfect match; checking only the winning chunk means an exclusion can
only ever cost a patent the exact evidence that made it look relevant
in the first place - a much narrower, more defensible check.
"""

from __future__ import annotations

import requests

from app.config import (
    RERANKER_REMOTE_API_KEY,
    RERANKER_REMOTE_BASE_URL,
    RERANKER_REMOTE_MODEL,
    RERANKER_REQUEST_TIMEOUT,
)
from app.query_understanding.models import ParsedQuery


def _validate_remote_score(value: float) -> float:
    """
    Guard against a remote relevance_score that violates the documented
    [0, 1] assumption this scale relies on - a defensive clamp for an
    out-of-contract value, not a rescale: an in-range score is returned
    unchanged.
    """
    if 0.0 <= value <= 1.0:
        return value
    clamped = min(1.0, max(0.0, value))
    print(
        f"[Reranker] Warning: remote relevance_score {value} is outside the "
        f"expected [0, 1] range - clamping to {clamped}."
    )
    return clamped


def _format_chunk(payload: dict | None) -> str:
    """Chunk text with its section prefixed, for structural context."""
    if not payload:
        return ""
    section = (payload.get("section") or "").strip()
    text = (payload.get("text") or "").strip()
    return f"Section: {section}\n{text}" if section else text


def _chunk_mentions(text: str, terms: list[str]) -> bool:
    """True if *text* literally contains any of *terms* (case-insensitive)."""
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms if term.strip())


class Reranker:
    """
    Rerank Qdrant search results using either the remote server or a local model.
    """

    def __init__(self):
        self.base_url = RERANKER_REMOTE_BASE_URL.rstrip("/")
        self.model = RERANKER_REMOTE_MODEL
        self.headers = {"Authorization": f"Bearer {RERANKER_REMOTE_API_KEY}"}

    def rerank(
        self,
        query: str,
        results: list,
        requirements: ParsedQuery | None = None,
    ) -> list[tuple[float, object]]:
        """
        Score every candidate chunk individually, then assign each
        candidate PATENT the MAX of its own chunks' scores - so a
        patent is judged by its single strongest piece of evidence
        among every chunk it has (see module docstring for where
        "every chunk" comes from), not by an arbitrary top-K subset or
        a length-punishing combined blob.

        Returns `(patent_score, chunk)` pairs, one per input chunk -
        every chunk of the same patent shares that patent's max score -
        sorted by score descending. *patent_score* is on a 0-10 scale.
        Each chunk's OWN individual score (also 0-10) is written to
        `chunk.payload["_chunk_relevance"]`, letting the aggregation
        step identify exactly which chunk earned the patent's score.

        *requirements*, when it carries exclusion terms (e.g. "without
        indium tin oxide"), drops a patent only if its OWN
        highest-scoring chunk - the evidence that earned it that score -
        literally names an excluded term (see module docstring for why
        only that one chunk, not the patent's whole chunk set, is
        checked).
        """

        if not results:
            return []

        texts = [_format_chunk(chunk.payload) for chunk in results]
        print(f"Reranking {len(texts)} chunks for query: {query}")
        raw_scores = self._score(query, texts)

        patent_max_score: dict[str, float] = {}
        patent_best_chunk: dict[str, object] = {}
        for chunk, raw in zip(results, raw_scores):
            patent_id = (chunk.payload or {}).get("patent_id")
            score10 = float(raw) * 10.0
            if chunk.payload is not None:
                chunk.payload["_chunk_relevance"] = score10
            if patent_id not in patent_max_score or score10 > patent_max_score[patent_id]:
                patent_max_score[patent_id] = score10
                patent_best_chunk[patent_id] = chunk

        exclusions = requirements.exclusions if requirements else []
        if exclusions:
            for patent_id, best_chunk in patent_best_chunk.items():
                text = (best_chunk.payload or {}).get("text", "")
                if _chunk_mentions(text, exclusions):
                    del patent_max_score[patent_id]

        scored = [
            (patent_max_score[(chunk.payload or {}).get("patent_id")], chunk)
            for chunk in results
            if (chunk.payload or {}).get("patent_id") in patent_max_score
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _score(self, query: str, texts: list[str]) -> list[float]:
        """
        Raw 0-1 relevance scores, aligned index-for-index with *texts*.
        """

        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/rerank",
            headers=self.headers,
            json={
                "model": self.model,
                "query": query,
                "documents": texts,
            },
            timeout=RERANKER_REQUEST_TIMEOUT,
        )
        print(f"Reranker response status code: {response.status_code}")
        response.raise_for_status()

        # The server's relevance_score is used as-is, under the
        # documented assumption (this endpoint's own example response
        # shape, [0, 1]-scaled) that it's already a comparable
        # relevance score. _validate_remote_score only guards against
        # a value that violates that assumption outright; it never
        # transforms an in-range score.
        scores = [0.0] * len(texts)
        for item in response.json()["results"]:
            scores[item["index"]] = _validate_remote_score(
                float(item["relevance_score"])
            )
        return scores

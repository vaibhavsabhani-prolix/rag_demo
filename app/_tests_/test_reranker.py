"""
Test cases for the simplified Reranker (app/reranker.py) and the
patent-aggregation logic it feeds (app/semantic_search.py).

The reranker scores EVERY candidate chunk individually via a single
cross-encoder call, then assigns each candidate patent the MAX of its
own chunks' scores - checking a patent's complete chunk set (however
many chunks it has - see SemanticSearch.search_detailed, which fetches
every indexed chunk of a candidate patent, not just a vector-search
top-K subset), not a similarity-biased handful. There is no hand-tuned
weighted blending left to test, so these cases focus on: a patent's
score being the max across all its chunks (even a "buried" one that
wouldn't have made a top-K cut), the chunk that earned that max being
recoverable for display, independent scoring across patents, sort
order, the exclusion hard-filter, and the patent-aggregation
sort/best-chunk-selection logic.

No model/network calls - `Reranker._score()` is stubbed throughout.

Run with:
    ./.venv/bin/python -m app._tests_.test_reranker
"""

import sys
import types

from app.query_understanding.models import ParsedQuery
from app.reranker import Reranker, _chunk_mentions, _format_chunk


# ==================================================================
# Builders
# ==================================================================


def _chunk(text: str, patent_id: str = "P1", chunk_id: int = 0, section: str = "DESCRIPTION"):
    return types.SimpleNamespace(
        id=f"{patent_id}-{chunk_id}",
        score=1.0,
        payload={
            "patent_id": patent_id,
            "chunk_id": chunk_id,
            "section": section,
            "text": text,
        },
    )


def _parsed(**kwargs) -> ParsedQuery:
    kwargs.setdefault("original_query", "q")
    kwargs.setdefault("semantic_query", "q")
    return ParsedQuery(**kwargs)


def _stub_reranker(score_fn):
    """A Reranker instance with __init__ skipped - _score is replaced with score_fn(query, texts)."""
    r = object.__new__(Reranker)
    r.use_remote = False
    r._score = score_fn
    return r


# ==================================================================
# Reranker.rerank()
# ==================================================================


def case_1_patent_score_is_max_of_its_chunks(_):
    """A patent's score is the highest of its own chunks' individual scores, not the first or an average."""

    def fake_score(query, texts):
        return [0.2, 0.9, 0.5]  # weak, strong, medium

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("weak passage", patent_id="P1", chunk_id=0),
        _chunk("strong passage", patent_id="P1", chunk_id=1),
        _chunk("medium passage", patent_id="P1", chunk_id=2),
    ]
    ranked = reranker.rerank("widget", chunks)
    scores = {score for score, _ in ranked}
    assert scores == {9.0}, f"expected the patent's max chunk score (0.9*10), got {scores}"


def case_2_buried_chunk_still_found(_):
    """A chunk that scores high but appears LAST in the input (as if beyond a top-K cut) still sets the patent's score - every chunk is checked, not just the first few."""

    def fake_score(query, texts):
        # 20 weak chunks, then one strong one at the very end.
        return [0.05] * 20 + [0.95]

    reranker = _stub_reranker(fake_score)
    chunks = [_chunk(f"weak passage {i}", patent_id="P1", chunk_id=i) for i in range(20)]
    chunks.append(_chunk("the actually relevant passage", patent_id="P1", chunk_id=20))

    ranked = reranker.rerank("widget", chunks)
    scores = {score for score, _ in ranked}
    assert scores == {9.5}, f"expected the buried chunk's score to win, got {scores}"


def case_3_different_patents_score_independently(_):
    """Two different patents' chunks are scored independently, not mixed up."""

    def fake_score(query, texts):
        return [0.9 if "genuinely relevant" in t else 0.1 for t in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("This patent is genuinely relevant to the query.", patent_id="P_GOOD"),
        _chunk("This patent is about something else entirely.", patent_id="P_BAD"),
    ]
    ranked = reranker.rerank("widget", chunks)
    by_patent = {chunk.payload["patent_id"]: score for score, chunk in ranked}
    assert by_patent["P_GOOD"] == 9.0
    assert by_patent["P_BAD"] == 1.0


def case_4_sorted_descending(_):
    """Results are sorted by score, highest first."""

    def fake_score(query, texts):
        return [0.3, 0.9, 0.1]

    reranker = _stub_reranker(fake_score)
    chunks = [
        _chunk("a", patent_id="PA"),
        _chunk("b", patent_id="PB"),
        _chunk("c", patent_id="PC"),
    ]
    ranked = reranker.rerank("widget", chunks)
    scores = [score for score, _ in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][1].payload["patent_id"] == "PB"


def case_5_score_scaled_to_ten(_):
    """The model's 0-1 output is scaled to a 0-10 relevance score."""

    def fake_score(query, texts):
        return [0.42]

    reranker = _stub_reranker(fake_score)
    chunks = [_chunk("some text")]
    ranked = reranker.rerank("widget", chunks)
    assert abs(ranked[0][0] - 4.2) < 1e-9


def case_6_each_chunk_own_relevance_stashed_on_payload(_):
    """Every chunk's OWN individual score (not just the patent max) is written to payload['_chunk_relevance'], so the best-evidence chunk can be identified later."""

    def fake_score(query, texts):
        return [0.2, 0.9]

    reranker = _stub_reranker(fake_score)
    weak = _chunk("weak passage", patent_id="P1", chunk_id=0)
    strong = _chunk("strong passage", patent_id="P1", chunk_id=1)
    reranker.rerank("widget", [weak, strong])

    assert abs(weak.payload["_chunk_relevance"] - 2.0) < 1e-9
    assert abs(strong.payload["_chunk_relevance"] - 9.0) < 1e-9


def case_7_empty_results_returns_empty(_):
    """No candidates in, nothing out - no model call attempted."""
    calls = []

    def fake_score(query, texts):
        calls.append(texts)
        return [0.5 for _ in texts]

    reranker = _stub_reranker(fake_score)
    assert reranker.rerank("widget", []) == []
    assert calls == []


def case_8_exclusion_drops_patent_when_its_winning_chunk_mentions_it(_):
    """A patent is dropped when the chunk that actually earned it its score literally names an excluded term."""

    def fake_score(query, texts):
        # The excluded-term chunk is also the highest-scoring one.
        return [0.9 if "indium tin oxide" in t else 0.5 for t in texts]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(exclusions=["indium tin oxide"])
    chunks = [
        _chunk("An OLED display comprising an indium tin oxide anode.", patent_id="P_EXCLUDED", chunk_id=0),
        _chunk("A weaker, unrelated passage from the same patent.", patent_id="P_EXCLUDED", chunk_id=1),
        _chunk("An OLED display, entirely unrelated passage.", patent_id="P_CLEAN", chunk_id=0),
    ]
    ranked = reranker.rerank("OLED display", chunks, requirements=parsed)
    patent_ids = {chunk.payload["patent_id"] for _, chunk in ranked}
    assert patent_ids == {"P_CLEAN"}, f"expected P_EXCLUDED dropped, got {patent_ids}"


def case_8b_exclusion_does_not_drop_patent_over_a_non_winning_chunk(_):
    """
    A patent survives when the excluded term only appears in a WEAKER
    chunk than the one that earned the patent its score - an incidental,
    out-of-context mention elsewhere in a long patent must not sink an
    otherwise genuine match. This is the actual bug this design fixes:
    Query Understanding can hallucinate an exclusion the user never
    asked for (e.g. inferring "not video" from a query about "still
    images"), and a long real patent will often mention an ordinary
    term like that somewhere unrelated to why it matched.
    """

    def fake_score(query, texts):
        # The chunk mentioning "video" scores low; the genuinely
        # relevant chunk (no mention of it) scores high and wins.
        return [0.2 if "video" in t else 0.9 for t in texts]

    reranker = _stub_reranker(fake_score)
    parsed = _parsed(exclusions=["video"])
    chunks = [
        _chunk(
            "Background: prior systems also handled video in unrelated ways.",
            patent_id="P1",
            chunk_id=0,
        ),
        _chunk(
            "The image storage device stores digital still images and prints them on command.",
            patent_id="P1",
            chunk_id=1,
        ),
    ]
    ranked = reranker.rerank("still image storage and printing", chunks, requirements=parsed)
    patent_ids = {chunk.payload["patent_id"] for _, chunk in ranked}
    assert patent_ids == {"P1"}, f"expected P1 to survive despite the unrelated 'video' mention, got {patent_ids}"
    scores = {score for score, _ in ranked}
    assert scores == {9.0}, f"expected the patent's score to be its winning chunk's score, got {scores}"


def case_9_no_exclusions_nothing_filtered(_):
    """Without exclusions (None or empty), every candidate patent is scored normally."""

    def fake_score(query, texts):
        return [0.9 for _ in texts]

    reranker = _stub_reranker(fake_score)
    chunks = [_chunk("mentions indium tin oxide anyway", patent_id="P1")]

    ranked_no_requirements = reranker.rerank("widget", chunks, requirements=None)
    assert len(ranked_no_requirements) == 1

    ranked_empty_exclusions = reranker.rerank(
        "widget", chunks, requirements=_parsed(exclusions=[])
    )
    assert len(ranked_empty_exclusions) == 1


def case_10_chunk_mentions_is_case_insensitive(_):
    assert _chunk_mentions("Contains Indium Tin Oxide here", ["indium tin oxide"])
    assert not _chunk_mentions("no match here", ["indium tin oxide"])
    assert not _chunk_mentions("some text", [])


def case_11_format_chunk_handles_missing_section(_):
    """A chunk with no section is formatted as bare text, no dangling 'Section:' prefix."""
    text = _format_chunk({"text": "bare text", "section": ""})
    assert text == "bare text"
    assert _format_chunk(None) == ""


# ==================================================================
# SemanticSearch._aggregate_by_patent()
# ==================================================================


def _searcher():
    from app.semantic_search import SemanticSearch

    searcher = SemanticSearch.__new__(SemanticSearch)
    searcher.db = types.SimpleNamespace(
        get_patents_metadata=lambda ids: {pid: {} for pid in ids}
    )
    return searcher


def _reranked_chunk(patent_id, chunk_id, text, chunk_relevance):
    return types.SimpleNamespace(
        id=f"{patent_id}-{chunk_id}",
        score=1.0,
        payload={
            "patent_id": patent_id,
            "chunk_id": chunk_id,
            "section": "DESC",
            "text": text,
            "_chunk_relevance": chunk_relevance,
        },
    )


def case_12_aggregation_picks_best_chunk_by_its_own_relevance(_):
    """Within a patent, the displayed best_chunk is the one whose OWN score matches the patent's max - not the first or last in the list."""
    weak = _reranked_chunk("P1", 0, "weaker passage", chunk_relevance=2.0)
    strong = _reranked_chunk("P1", 1, "stronger passage", chunk_relevance=8.0)

    # Patent-level score (both share it) is 8.0 - the max.
    reranked = [(8.0, weak), (8.0, strong)]

    results = _searcher()._aggregate_by_patent(reranked)
    assert len(results) == 1
    assert results[0].best_chunk.chunk_id == 1
    assert results[0].score == 8.0


def case_13_aggregation_sorts_topic_queries_by_score(_):
    """Topic queries (is_question=False) sort patents by relevance score, descending."""
    chunk_low = _reranked_chunk("P_LOW", 0, "low relevance", chunk_relevance=7.2)
    chunk_high = _reranked_chunk("P_HIGH", 0, "high relevance", chunk_relevance=9.5)

    reranked = [(7.2, chunk_low), (9.5, chunk_high)]

    results = _searcher()._aggregate_by_patent(reranked, is_question=False)
    assert [p.patent_id for p in results] == ["P_HIGH", "P_LOW"]


def case_14_aggregation_question_answer_first(_):
    """For question queries, a patent with a genuine extracted answer ranks above one with only a higher relevance score."""
    chunk_no_answer = types.SimpleNamespace(
        id="P_HIGH-0",
        score=1.0,
        payload={
            "patent_id": "P_HIGH",
            "chunk_id": 0,
            "section": "DESC",
            "text": "highly relevant but no direct answer found",
            "_chunk_relevance": 9.5,
        },
    )
    chunk_answered = types.SimpleNamespace(
        id="P_ANSWERED-0",
        score=1.0,
        payload={
            "patent_id": "P_ANSWERED",
            "chunk_id": 0,
            "section": "DESC",
            "text": "the sensor calibrates temperature using a lookup table",
            "_chunk_relevance": 7.1,
            "answer": "a lookup table",
            "answer_score": 0.9,
        },
    )

    reranked = [(9.5, chunk_no_answer), (7.1, chunk_answered)]

    results = _searcher()._aggregate_by_patent(reranked, is_question=True)
    assert [p.patent_id for p in results] == ["P_ANSWERED", "P_HIGH"]
    assert results[0].answer == "a lookup table"
    assert results[0].has_answer is True
    assert results[1].has_answer is False


def case_15_aggregation_empty_input(_):
    """No reranked chunks in, no patents out."""
    assert _searcher()._aggregate_by_patent([]) == []


CASES = [
    ("Test 1  - patent score is max of its chunks", case_1_patent_score_is_max_of_its_chunks),
    ("Test 2  - buried chunk (beyond a top-K cut) still found", case_2_buried_chunk_still_found),
    ("Test 3  - different patents score independently", case_3_different_patents_score_independently),
    ("Test 4  - results sorted descending", case_4_sorted_descending),
    ("Test 5  - score scaled to 0-10", case_5_score_scaled_to_ten),
    ("Test 6  - each chunk's own relevance stashed on payload", case_6_each_chunk_own_relevance_stashed_on_payload),
    ("Test 7  - empty results returns empty, no model call", case_7_empty_results_returns_empty),
    ("Test 8  - exclusion drops patent when its winning chunk mentions it", case_8_exclusion_drops_patent_when_its_winning_chunk_mentions_it),
    ("Test 8b - exclusion spares patent when only a non-winning chunk mentions it", case_8b_exclusion_does_not_drop_patent_over_a_non_winning_chunk),
    ("Test 9  - no exclusions, nothing filtered", case_9_no_exclusions_nothing_filtered),
    ("Test 10 - chunk_mentions is case-insensitive", case_10_chunk_mentions_is_case_insensitive),
    ("Test 11 - format_chunk handles missing section", case_11_format_chunk_handles_missing_section),
    ("Test 12 - aggregation picks best_chunk by its own relevance", case_12_aggregation_picks_best_chunk_by_its_own_relevance),
    ("Test 13 - aggregation sorts topic queries by score", case_13_aggregation_sorts_topic_queries_by_score),
    ("Test 14 - aggregation puts answered patent first for questions", case_14_aggregation_question_answer_first),
    ("Test 15 - aggregation empty input", case_15_aggregation_empty_input),
]


def main():
    failures = []
    for name, case in CASES:
        try:
            case(None)
            print(f"[PASS] {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failures.append(name)
            print(f"[ERROR] {name}: {e!r}")

    print()
    if failures:
        print(f"{len(failures)} of {len(CASES)} cases FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(CASES)} cases passed.")


if __name__ == "__main__":
    main()

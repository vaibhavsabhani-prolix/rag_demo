# Patent Search Pipeline — Query to Results, A to Z

This document explains what happens, in order, from the moment a user types a
query in [streamlit_app.py](streamlit_app.py) to the moment ranked patents are
shown on screen. It covers the current 7-phase search pipeline only. For how
patents get chunked, embedded, and indexed into Qdrant in the first place, see
[INDEXING_README.md](INDEXING_README.md).

The pipeline is orchestrated top-to-bottom in `streamlit_app.py` (search for
`Phase 1:` … `Phase 7:` around line 1478 onward). Each phase is a standalone
module under `app/`, takes the previous phase's output as input, and produces
a typed Pydantic result object that both feeds the next phase and renders its
own section in the UI.

```text
User types a query
       │
       ▼
Phase 1  Query Understanding            (LLM call, cached)          → ParsedQuery
       │
       ▼
Phase 2  Candidate Retrieval            (vector search, Qdrant)      → CandidateRetrievalResult
       │
       ▼
Phase 3  Metadata Filtering             (deterministic, no I/O)      → FilteredCandidateResult
       │
       ▼
Phase 4  Bounded Evidence Retrieval     (vector search, Qdrant)      → EvidenceRetrievalResult
       │
       ▼
Phase 5  Relationship/Requirement Verification (cross-encoder)       → VerificationBatchResult
       │
       ▼
Phase 6  BGE Cross-Encoder Reranking    (cross-encoder)              → RerankBatchResult
       │
       ▼
Phase 7  Final Scoring & Selection      (deterministic, no I/O)      → FinalSearchResult
       │
       ▼
Ranked patent cards on screen
```

Only two things ever call an external model over the network: **Phase 1**
(one LLM call) and **Phase 5 + Phase 6** (a shared BGE cross-encoder server).
Every other phase is pure Python running against data already in memory or
in Qdrant.

---

## Phase 1 — Query Understanding

**Module:** `app/query_understanding/engine.py` (`QueryUnderstandingEngine.parse`)
**Config:** `QUERY_LLM_REMOTE_BASE_URL`, `QUERY_LLM_REMOTE_MODEL`, `QUERY_CACHE_SIZE`, `QUERY_LLM_TEMPERATURE`, `QUERY_LLM_MAX_TOKENS`, `QUERY_LLM_REQUEST_TIMEOUT`

The raw query string is turned into a structured `ParsedQuery` object — this
is the single point where natural language gets converted into something the
rest of the pipeline can reason about mechanically.

```text
PHASE 1 — QUERY UNDERSTANDING

User Query (raw text)
        ↓
  Check LRU Cache
        ↓
┌───────┴───────┐
↓               ↓
Cache HIT       Cache MISS
↓               ↓
Return cached   Build prompt
ParsedQuery     (system + 2 few-shot anchor examples + query)
                ↓
                Call remote LLM (temperature = 0, max_retries = 1)
                ↓
                Parse JSON response
                ↓
          ┌─────┴─────┐
          ↓           ↓
       Valid      Truncated
       JSON       (finish_reason=length)
          ↓           ↓
          │      Salvage every field the model
          │      finished generating (repair
          │      partial JSON, drop incomplete
          │      trailing objects)
          └─────┬─────┘
                ↓
                Validate against ParsedQuery schema
                ↓
                Local deterministic normalization
                ↓
                Store result in cache
└───────┬───────┘
        ↓
  PHASE 1 OUTPUT

  ParsedQuery
  (metadata_filters, concepts, relationships,
   requirements, semantic_query, is_metadata_only)
        ↓
  Send to Phase 2
```

1. **Cache check** (`app/query_understanding/cache.py`) — an in-memory LRU
   cache keyed on the exact query string. A cache hit skips everything below
   and returns instantly (shown in the UI as "⚡ CACHE HIT").
2. **Warm-up on startup** — `QueryUnderstandingEngine.warm_up()` fires one
   throwaway request at the remote LLM when the engine singleton is created
   (`streamlit_app.py`'s `get_engine()`), so a cold/unloaded remote model
   pays its startup latency once at app launch instead of on a real user's
   first query.
3. **One LLM call** — the query is sent to a remote OpenAI-compatible LLM
   (`QUERY_LLM_REMOTE_MODEL`, temperature `0.0` for determinism) with a system
   prompt (`app/query_understanding/prompt.py`) plus **two** few-shot anchor
   examples (a short simple query, and a dense multi-field query) asking it to
   return structured JSON, not prose. `response_format={"type": "json_object"}`
   forces valid JSON back.
4. **JSON truncation repair** (`_repair_truncated_json` in `engine.py`) — if
   the response is cut off mid-generation (a very information-dense query can
   need more output than `QUERY_LLM_MAX_TOKENS` allows), this walks the raw
   text and salvages every field the model finished generating instead of
   discarding the whole response. It only accepts a cut point that leaves
   valid, semantically-complete JSON (e.g. it will drop a relationship object
   that got cut off mid-key rather than keep a `{"subject": "..."}` with no
   `relation`/`object`), so `ParsedQuery`'s safe field defaults absorb
   whatever didn't make it.
5. **Schema validation** — the JSON is parsed into a `ParsedQuery`
   (`app/models/parsed_query.py`), which has:
   - `metadata_filters` — structured `(field, operator, value)` constraints (assignee, date, CPC/IPC code, country, etc.).
   - `concepts` — up to 8 of the most essential technical concepts/entities mentioned.
   - `relationships` — directed triples like `(manufacturing method, produces, water)`.
   - `attributes` — concept/property/value triples.
   - `requirements` — plain-English sentences the patent must satisfy.
   - `constraints` / `exclusions` — numeric/negative constraints.
   - `semantic_query` — a *short* (≤~30 word) intent-preserving paraphrase of the query.
   - `is_metadata_only` — true only when the query is pure metadata with no technical content (e.g. *"patents filed by Samsung in 2020"*).
6. **Local normalization** (`app/query_understanding/normalizer.py`) —
   deterministic cleanup (no LLM call) of extracted values, including CPC/IPC
   code punctuation stripping (see the callout below).
7. **Cache write** — the normalized result is cached for next time.
8. **Failure handling** — any LLM/schema error returns a conservative
   fallback `ParsedQuery` (just `semantic_query` = the raw query, everything
   else empty) rather than crashing the pipeline.

> **Past bugs that lived here:**
> - The JSON schema used to list `semantic_query` *first* and describe it as
>   `"<complete intent query>"`. For a dense, fact-heavy query, the model read
>   that literally and just copied the entire input into `semantic_query`,
>   leaving every structured field empty — there was no example showing it
>   what to do with a long, multi-field query. Fixed by moving
>   `metadata_filters` first in the schema (so it's generated before the
>   model can run out of token budget), capping `concepts` at 8 items instead
>   of exhaustively enumerating every noun phrase (which was itself eating
>   the budget meant for `metadata_filters`), and adding a second few-shot
>   anchor demonstrating extraction from a dense, many-field query.
> - Classification codes: the model sometimes "corrects" a bare code like
>   `H04N7163` into standard notation `H04N7/163`, but Qdrant's CPC/IPC
>   payload fields store the unpunctuated form — so an exact-match filter
>   silently failed. `QueryNormalizer.normalize_value` now strips
>   `/`, `-`, `.`, and spaces from CPC/IPC values regardless of how the LLM
>   chose to format them.
> - A response cut off by `max_tokens` used to `json.loads()`-fail entirely,
>   discarding every field the model *did* finish generating (including a
>   fully-populated `metadata_filters`) in favor of an empty fallback. The
>   truncation-repair step above fixes this generally, not per-query.

Rendered in the UI as **Phase 1: Query Understanding**, including the raw
JSON the LLM produced.

---

## Phase 2 — Dynamic Candidate Retrieval (Vector Search)

**Module:** `app/retrieval/retriever.py` (`CandidateRetriever.retrieve_candidates`)
**Config:** `RETRIEVAL_TOP_K_PER_VIEW=500`, `PATENT_CANDIDATE_TOP_K=300`

This phase asks Qdrant "which patents are even plausibly relevant?" — cheap,
broad, embedding-similarity-based, and intentionally over-inclusive (later
phases narrow it down).

```text
PHASE 2 — CANDIDATE RETRIEVAL

           ParsedQuery
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
 Original   Semantic   Structured
    ↓          ↓          ↓
 Qdrant     Qdrant     Qdrant
    ↓          ↓          ↓
 100 hits   100 hits   100 hits
    └──────────┼──────────┘
               ↓
        Merge duplicates
               ↓
        Group by patent
               ↓
    Best chunk = patent score
               ↓
        Sort patents
               ↓
        Keep top 300
               ↓
  PHASE 2 OUTPUT

  CandidateRetrievalResult
  (up to 300 candidate patents,
   each with its matched chunks + best score)
               ↓
        Send to Phase 3
```

**Metadata-only branch:** if `parsed_query.is_metadata_only`, vector search
is skipped entirely — patents are matched purely by scrolling the metadata
collection (`app/retrieval/filter_builder.py`) and returned with a placeholder
retrieval score of `1.0`.

**Semantic branch** (the normal case):

1. **Build multiple retrieval "views"** (`app/retrieval/views.py`,
   `build_retrieval_views`) — deterministic, no LLM call. Up to three text
   variants of the same query are built so different phrasings of the same
   intent can each surface their own best matches:
   - `original` — the raw user query.
   - `semantic` — the Phase 1 `semantic_query`.
   - `structured` — a compact synthesis of concepts, relationships,
     attributes, and requirements (e.g. `"Relationships:\nmanufacturing method produces water"`).
2. **Batch-embed all views in one HTTP call** (`app/embedder.py`,
   `embed_texts`) — one round trip regardless of how many views there are.
3. **Optional metadata pre-filter (best-effort, not a hard gate)** — if
   `metadata_filters` are present, the patent-ID set is narrowed first (via
   the metadata collection), and the vector search below is restricted to
   only those patent IDs *if* the narrowing found at least one match. If it
   finds **zero** matching patent IDs, the search does **not** short-circuit
   to an empty result — it falls back to an unrestricted vector search
   across the whole collection, and lets Phase 3's stricter, more forgiving
   metadata check (below) do the real enforcement on whatever candidates
   come back. See the callout below for why.
4. **Vector search per view** — each view's embedding is searched against the
   `CHUNKS_COLLECTION_NAME` collection independently, returning up to
   `RETRIEVAL_TOP_K_PER_VIEW` (500) chunk hits per view.
5. **Deduplicate and merge** — hits from different views that land on the
   same chunk are merged (keeping the max score, recording which views
   matched it).
6. **Group into candidate patents** — chunks are grouped by `patent_id`; each
   patent's `retrieval_score` is its single best-scoring chunk.
7. **Bound the pool** — patents are sorted by that score and truncated to
   `PATENT_CANDIDATE_TOP_K` (300).
8. **Enrich with metadata** — title, assignee, dates, etc. are batch-fetched
   from the metadata collection for the surviving 300 candidates.

> **Past bug that lived here:** the metadata pre-filter used to be a hard
> gate — if it found zero matching patent IDs (for *any* reason: a wrong
> field-code mapping, an LLM-invented constraint the resolver couldn't
> place, a formatting mismatch), the whole search returned zero results
> immediately, and the vector search that would have found the patent by
> its actual content never ran at all. Since metadata extraction/mapping is
> exactly the kind of thing that can have edge cases, this made the entire
> search fragile to a single bad filter. It's now explicitly an
> optimization: narrow the search when the narrowing itself is trustworthy
> (found something), otherwise fall back to the same broad search a
> filter-less query would get, and let Phase 3 apply the real, per-candidate
> metadata check afterward.

Rendered as **Phase 2: Candidate Retrieval**, showing per-view hit counts and
the merged candidate list.

---

## Phase 3 — Metadata Filtering & Constraint Enforcement

**Module:** `app/retrieval/metadata_filter.py` (`filter_candidates`)

A purely deterministic, in-memory pass — no network calls. This is a second,
stricter metadata check on the 300 candidates that survived Phase 2 (Phase
2's own metadata narrowing, if any, only pruned which patent IDs vector
search was allowed to look at — this phase re-validates every field on the
actual candidate objects).

```text
PHASE 3 — METADATA FILTERING

  Phase 2 Output (≤ 300 candidates)
               ↓
    metadata_filters present?
    ┌──────────┴──────────┐
    ↓                     ↓
   No                    Yes
    ↓                     ↓
 Pass all           For each candidate patent:
 through             check EVERY filter (AND)
 unchanged                ↓
                    Date/Year field?  → numeric compare
                    Code field (CPC/IPC)? → prefix/exact match
                    Other field? → normalized text match
                          ↓
                    Missing required field → FAIL
                          ↓
                    All filters passed?
                    ┌─────┴─────┐
                    ↓           ↓
                  Keep         Drop
                    └─────┬─────┘
    └──────────┬──────────┘
               ↓
     Record per-filter pass/fail diagnostics
               ↓
  PHASE 3 OUTPUT

  FilteredCandidateResult
  (surviving candidates + per-filter diagnostics)
               ↓
        Send to Phase 4
```

- Every `MetadataFilter` from Phase 1 is checked with **strict AND
  semantics** — a candidate must pass *all* filters to survive.
- Field comparison is type-aware:
  - **Date/year fields** (`PY`, `PD`, `AD`, or any field name containing
    "year"/"date") — numeric comparison, "smart" about comparing a year
    against a full date; `in`/`any` (list membership) is supported the same
    as `==` here.
  - **Code fields** (`CPC`, `IPC`, `PNC`, `AC`, or any field containing
    "cpc"/"ipc"/"country"/"jurisdiction") — normalized (punctuation
    stripped) exact/prefix match.
  - **Name/free-text fields** (inventor, assignee, applicant, title, address)
    — punctuation-insensitive comparison, so a dataset name stored as
    `"Last, First"` matches a query written as `"First Last"`.
  - **Everything else** — normalized case-insensitive text match/substring/contains.
- **Missing metadata is a hard fail** for `==`-style operators — but only
  when the field *resolved to a real, known payload key* and that key is
  genuinely absent from the candidate. A field name that never resolved to
  any known key at all (`resolve_payload_field_names` in
  `app/retrieval/filter_builder.py`, shared with Phase 2's pre-filter,
  returns `[]`) is treated as **unverifiable, not failed** — see the callout
  below for why that distinction matters. (`!=` filters trivially pass when
  the field is absent, resolved or not.)
- Per-filter pass/fail counters are recorded for the **Diagnostics** panel,
  so it's visible in the UI exactly which filter, if any, is eliminating
  candidates.

If `parsed_query.metadata_filters` is empty, this phase is a no-op that
passes every candidate through unchanged.

> **Past bugs that lived here** (in the shared `filter_builder.py` field
> resolution used by both this phase and Phase 2's pre-filter):
> - An unresolved field name used to fall back to guessing the literal
>   string (or a generic reference-table description) as the Qdrant payload
>   key. That guess is almost never right, and a resolvable field silently
>   "not found" and a genuinely-unknown field were treated identically
>   (both = hard fail) — so one LLM-invented or mismapped field name (e.g.
>   `cpc_12_digit`, or `Attorney/Agent English` guessed from a generic
>   table when the real key was just `Attorney/Agent`) could zero out an
>   entire query that had 30 other, perfectly correct filters. Now:
>   resolvable-but-absent fails (a real mismatch), unresolvable is skipped
>   (can't verify it either way, so don't punish the candidate for it).
> - Name fields used to get a hard Qdrant-level exact-match/`MatchAny`
>   condition at the Phase 2 pre-filter stage. This dataset has no full-text
>   payload index on those fields, so Qdrant's `MatchText` silently matched
>   nothing instead of erroring — every inventor/assignee filter was
>   effectively dead on arrival at the Qdrant level, regardless of how the
>   *name itself* was formatted. Name fields are never given a hard Qdrant
>   condition now; they're deferred entirely to this phase's Python-side
>   comparison.
> - Year/date fields normalized to `int` by Phase 1 (for range comparisons)
>   used to be passed straight into Qdrant's exact-match condition — but
>   this dataset stores them as **strings** (`"2006"`, not `2006`), and
>   `MatchValue` is type-sensitive. Every exact year filter silently failed
>   until values were always compared as strings for equality.
> - `in`/`any` (list membership) on a date/year field used to fall through
>   every branch of the date comparator with no matching case, silently
>   returning `False` regardless of whether the data matched — e.g.
>   `"priority years 2005 and 2006"` on a patent whose actual `Priority Year`
>   was exactly `[2005, 2006]` still failed.

Rendered as **Phase 3: Metadata Filtering & Constraint Enforcement**.

---

## Phase 4 — Bounded Evidence Retrieval

**Module:** `app/retrieval/evidence_retriever.py` (`EvidenceRetriever.retrieve_evidence`)
**Config:** `EVIDENCE_CHUNKS_PER_PATENT=5`, `EVIDENCE_NEIGHBOR_CHUNKS=1`, `EVIDENCE_GLOBAL_TOP_K_CHUNKS=1000`

Phase 2 only surfaced whichever chunk happened to score best per view — this
phase fetches a small, focused bundle of the *most relevant* text per
surviving candidate, which downstream verification/reranking will actually
read.

```text
PHASE 4 — BOUNDED EVIDENCE RETRIEVAL

  Phase 3 Output (surviving candidates)
               ↓
  Build ONE evidence query
  (semantic_query + relationships + requirements + concepts)
               ↓
  Seed pool with chunks Phase 2 already fetched
  (retrieval_source = "initial_candidate")
               ↓
  Embed evidence query (one HTTP call)
               ↓
  Vector search — restricted to surviving candidates' patent_ids
  (bounded to candidates × chunks_per_patent × 2,
   capped at EVIDENCE_GLOBAL_TOP_K_CHUNKS)
               ↓
  Merge into pool
  (retrieval_source = "evidence_query")
               ↓
  Fetch neighbor chunks (±1) in ONE batched scroll
  (retrieval_source = "neighbor")
               ↓
  Sort chunks per patent:
  direct evidence first, then neighbors, by score
               ↓
  Keep top 5 chunks per patent
               ↓
  PHASE 4 OUTPUT

  EvidenceRetrievalResult
  (each surviving patent → up to 5 evidence chunks)
               ↓
        Send to Phase 5
```

1. **Build one evidence query** (`build_evidence_query`) — deterministic
   concatenation of `semantic_query`, relationships, requirements, and
   concepts (no LLM call).
2. **Seed with what Phase 2 already fetched** — chunk text already in hand
   from Phase 2 goes in first (`retrieval_source="initial_candidate"`).
3. **One more vector search**, restricted by `patent_id` to only the
   surviving candidates, using the evidence query's embedding — bounded to
   `min(candidates × chunks_per_patent × 2, EVIDENCE_GLOBAL_TOP_K_CHUNKS)`
   total hits (`retrieval_source="evidence_query"`).
4. **Pull neighbor chunks** — for every chunk collected so far, its immediate
   textual neighbors (±`EVIDENCE_NEIGHBOR_CHUNKS`) are fetched in one batched
   Qdrant `scroll` call, so a matching chunk's surrounding context isn't lost
   mid-sentence (`retrieval_source="neighbor"`, scored slightly below its anchor).
5. **Bound per patent** — chunks are sorted (direct evidence before
   neighbors, by score) and truncated to `EVIDENCE_CHUNKS_PER_PATENT` (5) per patent.

Every patent entering Phase 5 carries at most 5 chunks of text — enough for
the cross-encoder stages to judge it without shipping a patent's entire
(potentially thousands-of-chunks) document.

Rendered as **Phase 4: Bounded Evidence Retrieval**.

---

## Phase 5 — Semantic Relationship & Requirement Verification

**Module:** `app/verification/verifier.py` (`RelationshipVerifier.verify_candidates`)
**Config:** `VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD=0.35`, `VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD=0.35`, `VERIFICATION_MIN_EVIDENCE_SCORE=0.15`

**No candidate cap** — every candidate that survived Phase 2/3 gets verified
here, not just a top-N slice by vector-similarity score (there used to be a
`VERIFICATION_MAX_CANDIDATES=25` cap; see the callout below for why it was
removed). This means Phase 5/6 cost scales with how many candidates survive
metadata filtering — a broad, filter-less query with nothing narrowing the
pool can mean verifying/reranking the full 300-candidate pool, which is
noticeably slower than a filter-narrowed query but never silently drops a
real match by rank.

This is the **precision gate** — the phase responsible for deciding, patent
by patent, whether the query's specific relationships and requirements are
actually backed by evidence text, as opposed to the patent merely sharing
generic vocabulary with the query.

> This is also where a past bug lived: the support threshold used to be a
> hardcoded `0.05`, which is far too permissive for a BGE cross-encoder —
> near-zero relevance scores would still count as "supported". That let
> patents which only share generic words with the query (e.g. "method",
> "manufacturing") get marked as satisfying a specific relationship (e.g.
> "produces water") even when the actual subject never appeared in the
> evidence. Both thresholds are now real config values in `app/config.py`
> instead of magic numbers, specifically so this can be tuned without
> touching code.

```text
PHASE 5 — RELATIONSHIP & REQUIREMENT VERIFICATION

  Phase 4 Output (candidates + up to 5 evidence chunks each)
               ↓
    is_metadata_only OR no relationships/requirements?
    ┌──────────┴──────────┐
    ↓                     ↓
   Yes                   No
    ↓                     ↓
 Auto-pass          Take ALL surviving candidates (no cap)
 every patent               ↓
 (coverage = 1.0)   Build one hypothesis string
                    per relationship
                    ("subject relation object" — plain,
                     no context clause appended)
                          ↓
                    Build one hypothesis string
                    per requirement
                          ↓
                    Cross-encoder scores:
                    hypothesis × every evidence chunk
                          ↓
                    Span-proximity check
                    (subject & object words within
                     40 words of each other?)
                          ↓
                    score ≥ 0.35?
                    ┌──────┴──────┐
                    ↓             ↓
                   Yes            No
                    ↓             ↓
               SUPPORTED    proximity AND score ≥ 0.15?
                             ┌──────┴──────┐
                             ↓             ↓
                            Yes            No
                             ↓             ↓
                        SUPPORTED    NOT_SUPPORTED
                    └──────┬──────┘
                           ↓
                    relationship_coverage = supported / total
                    requirement_coverage  = supported / total
    └──────────┬──────────┘
               ↓
  PHASE 5 OUTPUT

  VerificationBatchResult
  (coverage ratios + per-relationship/requirement verdicts)
               ↓
        Send to Phase 6
```

Skipped entirely (auto-pass, coverage = 1.0) when the query is metadata-only
or carries no relationships/requirements — there's nothing to verify.

Otherwise, for **every** surviving candidate:

1. **Build one hypothesis string per relationship** — just
   `"{subject} {relation} {object}"` (e.g. `"storage stores water"`), no
   context clause appended — see the callout below for why.
2. **Build one hypothesis string per requirement** — the requirement's own text.
3. **Batch cross-encoder scoring** — every hypothesis is scored against every
   evidence chunk of every candidate in batched calls to the same BGE
   reranker server used in Phase 6 (`_score_pairs_remote`).
4. **Deterministic span-proximity check** (`check_span_proximity`) — a
   cheap, non-ML backstop: do the relationship's subject words and object
   words literally co-occur within 40 words of each other in a chunk?
5. **Relationship verdict** — `SUPPORTED` if the cross-encoder score alone
   clears `VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD` (0.35), **or** the
   span-proximity check fires *and* the score also clears the much lower
   `VERIFICATION_MIN_EVIDENCE_SCORE` (0.15) floor; `NOT_SUPPORTED` otherwise
   — see the callout below for why proximity alone is no longer sufficient.
6. **Requirement verdict** — similar, but blends the cross-encoder score with
   a word-overlap ratio (`overlap * 0.5`) as a fallback signal, checked
   against `VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD`. This word-overlap
   fallback has the same class of weakness the relationship check used to
   have (a loose heuristic that can independently manufacture a match) and
   has not yet been tightened the same way — worth revisiting if a query's
   requirement is getting satisfied by content that doesn't actually satisfy it.
7. **Coverage ratios** — `relationship_coverage` and `requirement_coverage`
   are simply `supported / total` for that patent. A patent with an explicit
   `CONTRADICTED` relationship status is later zeroed out in Phase 7
   regardless of its raw coverage number.

> **Past bugs that lived here:**
> - `is_supported` used to be `has_proximity OR score >= threshold` — an
>   unconditional `OR`. `check_span_proximity` is a crude, whole-chunk
>   word-distance check with no sense of semantic role, so it could fire on
>   pure coincidence (e.g. "water" and "storage" both appearing in a
>   multi-thousand-word chunk, in a sentence about an unrelated liquid-level
>   *sensor*, not about water being stored) and completely override a
>   cross-encoder score that had correctly judged the chunk as ~0% relevant.
>   Real-world result: an LNG storage tank patent and a cryogenic fluid
>   tank patent both ranked #1/#3 for a "water storage" query, neither
>   patent mentioning water as the stored substance at all. Fixed by
>   requiring the cross-encoder score to clear a floor
>   (`VERIFICATION_MIN_EVIDENCE_SCORE`) whenever proximity is the deciding
>   factor — proximity can now only boost an already-plausible score, never
>   manufacture one from ~0 relevance.
> - The relationship hypothesis used to append `" in {context}"` (e.g.
>   `"storage stores water in water-only storage"`). Measured against an
>   identical, clearly-matching chunk, this scored the cross-encoder ~6x
>   lower (`0.07` vs `0.42`) than the plain `"storage stores water"` —  the
>   context field often restates subject/object words in a way the model
>   reads as redundant/unnatural rather than clarifying. Dropped entirely.
> - A compound object like `"LED display"` tokenizes to `{"led", "display"}`,
>   and `check_span_proximity` only required *any one* of those words to be
>   nearby the subject — so the generic word "display" (present in nearly
>   every television patent) alone satisfied it, regardless of whether "LED"
>   ever appeared anywhere in the document. This is a known remaining gap:
>   a stricter fix (requiring a multi-word term's own words to cluster
>   together, plus acronym/expansion normalization for terms like
>   LCD↔"liquid crystal") was prototyped in-session but is not currently on
>   disk — check `app/verification/verifier.py` for `all_key_terms_present`
>   before relying on this being fixed.

These two coverage numbers matter a lot downstream — together they make up
**70% of the final score** in Phase 7 (`FINAL_WEIGHT_RELATIONSHIP` +
`FINAL_WEIGHT_REQUIREMENT` = 0.45 + 0.25), so getting the support threshold
right here is the single highest-leverage knob for search precision.

Rendered as **Phase 5: Semantic Relationship Verification**.

---

## Phase 6 — BGE Cross-Encoder Reranking

**Module:** `app/reranking/reranker.py` (`BGEReranker.rerank_candidates`)
**Config:** `RERANKER_REMOTE_BASE_URL`/`_MODEL` (`BAAI/bge-reranker-v2-m3`), `RERANK_BATCH_SIZE=128`, `RERANK_CONCURRENT_REQUESTS=6`, `RERANKER_MAX_CONTEXT_TOKENS=4096`

Where Phase 5 asks "is the query's *structure* (relationships/requirements)
satisfied?", Phase 6 asks a simpler, complementary question: "how relevant
is this evidence text to the query overall?" — a general cross-encoder
relevance score, independent of the structured verification above.

```text
PHASE 6 — BGE CROSS-ENCODER RERANKING

  Phase 5 Output (verified candidates)
               ↓
  Build ONE reranking query
  (semantic_query + relationships + requirements)
               ↓
  For every evidence chunk of every candidate:
  truncate to fit 4096-token budget
  (query tokens + doc tokens + safety margin)
               ↓
  Group chunks into batches of 128
               ↓
  Fire up to 6 batches concurrently (thread pool)
               ↓
  Cross-encoder scores: chunk vs reranking query
               ↓
  best_reranker_score = MAX(chunk scores) per patent
  avg_reranker_score  = AVG(chunk scores) per patent
               ↓
  PHASE 6 OUTPUT

  RerankBatchResult
  (best/avg reranker score per patent;
   Phase 5 verification data carried forward unchanged)
               ↓
        Send to Phase 7
```

1. **Build one reranking query string** (`build_rerank_query`) — semantic
   query + relationships + requirements, deterministically concatenated.
2. **Token-budget every chunk** (`truncate_document_for_budget`) — each
   chunk (with its `Section: X` prefix if it has one) is truncated so
   `query_tokens + doc_tokens + safety_margin` never exceeds the reranker's
   hard `4096`-token context limit. A `400` "token limit" response triggers
   one automatic retry with more aggressive truncation.
3. **Batched, concurrent scoring** — every evidence chunk (from every
   surviving candidate) is scored against the reranking query, `128` chunks
   per HTTP request, up to `6` requests in flight at once via a thread pool.
4. **Per-patent aggregation** — a patent's `best_reranker_score` is the max
   score across its own chunks (its single strongest piece of evidence);
   `avg_reranker_score` is also kept for visibility.
5. **Resilient to failure** — if every reranker endpoint fails, chunks fall
   back to a `0.0` score rather than crashing the whole search.

All of Phase 5's verification data (relationships, requirements, coverage
counts) is carried forward unchanged into `RerankedPatentResult` — Phase 6
only adds the cross-encoder score, it doesn't recompute or discard anything.

Rendered as **Phase 6: BGE Cross-Encoder Reranking**.

---

## Phase 7 — Final Patent Scoring & Result Selection

**Module:** `app/scoring/scorer.py` (`FinalScorer.score_and_rank`)
**Config:** `FINAL_SCORE_THRESHOLD`, `FINAL_WEIGHT_RELATIONSHIP=0.45`, `FINAL_WEIGHT_REQUIREMENT=0.25`, `FINAL_WEIGHT_RERANKER=0.20`, `FINAL_WEIGHT_RETRIEVAL=0.10`

The final phase is pure arithmetic — no model calls, no I/O — that combines
every signal collected so far into one 0–10 score per patent, then decides
what actually gets shown.

```text
PHASE 7 — FINAL PATENT SCORING & RESULT SELECTION

  Phase 6 Output (reranked + verified candidates)
               ↓
  For each candidate, compute 4 components (0.0 – 1.0):
    relationship_score   (forced to 0 if CONTRADICTED)
    requirement_score
    reranker_score
    retrieval_score
               ↓
  Weighted sum × 10:
    0.45 × relationship_score
  + 0.25 × requirement_score
  + 0.20 × reranker_score
  + 0.10 × retrieval_score
               ↓
  final_score (0 – 10)
               ↓
  final_score ≥ FINAL_SCORE_THRESHOLD?
  ┌───────────┴───────────┐
  ↓                        ↓
 Yes                       No
  ↓                        ↓
Qualify                  Reject
  └───────────┬───────────┘
               ↓
  Sort qualifying patents descending by final_score
               ↓
  PHASE 7 OUTPUT

  FinalSearchResult
  (ALL qualifying patents, no fixed top-N cap,
   each with a full ScoreBreakdown for explainability)
               ↓
  Rendered as ranked patent result cards
```

For every candidate that made it through Phase 6:

1. **Four normalized (0.0–1.0) component scores:**
   - `relationship_score` = Phase 5's `relationship_coverage`, forced to
     `0.0` if the patent had **any** `CONTRADICTED` relationship — an
     explicit contradiction overrides whatever coverage ratio it otherwise had.
   - `requirement_score` = Phase 5's `requirement_coverage`.
   - `reranker_score` = Phase 6's `best_reranker_score`.
   - `retrieval_score` = Phase 2's original `candidate_score` (the initial
     embedding-similarity signal, carried all the way through).
2. **Weighted sum on a 0–10 scale:**
   ```text
   final_score = 10 × ( 0.45 × relationship_score
                       + 0.25 × requirement_score
                       + 0.20 × reranker_score
                       + 0.10 × retrieval_score )
   ```
   Weights must sum to exactly `1.0` (`validate_weights`, checked at
   `FinalScorer.__init__` — raises immediately if `app/config.py` is
   misconfigured).
3. **Filter** — only patents with `final_score >= FINAL_SCORE_THRESHOLD`
   qualify. Every patent is still counted (`passed_threshold_count`,
   `rejected_count`) even if it doesn't make the cut, for the diagnostics panel.
4. **Sort** descending by `final_score`. **All** qualifying patents are
   returned — there is no fixed top-N cap; the score threshold alone decides
   how many results are shown.
5. **Full explainability** — every result carries a `ScoreBreakdown` (each
   component's raw value and its weighted contribution) plus every Phase
   5/6 relationship, requirement, and evidence chunk that produced it, so
   the UI can show exactly *why* a patent scored what it did.

Rendered as **Phase 7: Final Patent Scoring & Result Selection**, followed by
the patent result cards themselves.

---

## Why a patent can rank highly for the wrong reason (and how to fix it)

Because 70% of the final score comes from Phase 5's coverage ratios, and
those ratios are threshold-gated cross-encoder judgments, the single biggest
lever for search precision is
`VERIFICATION_RELATIONSHIP_SUPPORT_THRESHOLD` / `VERIFICATION_REQUIREMENT_SUPPORT_THRESHOLD`
in `app/config.py`:

- **Too low** → patents that only share generic vocabulary with the query
  (not its actual subject) get marked `SUPPORTED` and can out-rank truly
  relevant results.
- **Too high** → genuinely relevant patents that use different wording than
  the query get marked `NOT_SUPPORTED` and disappear from results entirely.

The second lever is `VERIFICATION_MIN_EVIDENCE_SCORE` — it controls how much
the deterministic span-proximity check is allowed to override the
cross-encoder. Too high and proximity effectively never helps (relying on the
cross-encoder alone); too low and proximity drifts back toward its old
failure mode of manufacturing matches from near-zero relevance (see the Phase
5 callout above).

Even with both thresholds well-tuned, the verification layer only checks
"do the subject and object words show up near each other / does the
cross-encoder think this is relevant" — it does not verify that a
multi-word qualifier (LED, wireless, lithium, ...) actually describes the
*specific thing* the query asked about, as opposed to some other, unrelated
component that happens to share one word with it. A patent titled
"Multi-functional **LCD** TV" with a separate, unrelated **LED** indicator
lamp can still satisfy a naive "television has LED display" check, because
"LED" and "display" are each present in the document without ever
describing the same component. Treat a `SUPPORTED` relationship as "the
query's words are grounded in this evidence," not as "this exact claim is
true," especially for compound technical terms.

The next lever is `FINAL_SCORE_THRESHOLD` in Phase 7 — it decides how many
of the qualifying patents actually get shown at all, independent of how they
were scored.

Finally, Phase 5 no longer caps how many candidates get verified/reranked —
every candidate that survives Phase 2/3 gets the full treatment. This trades
latency for recall: a broad, filter-less query with nothing narrowing the
candidate pool can mean verifying and reranking the full 300-candidate pool
(tens of seconds), where a filter-narrowed query stays fast. There is
currently no cap to re-add if that latency becomes a problem — see
`RelationshipVerifier.verify_candidates`'s `max_candidates` parameter, which
still exists and defaults to `None` (no limit), if you need to reintroduce one.

---

## Where to look for each config value

All tunables referenced above live in one place: [app/config.py](app/config.py).
The comment directly above each constant explains what raising or lowering
it does — that's the intended way to tune behavior; none of these values are
duplicated or hardcoded elsewhere in the pipeline modules.

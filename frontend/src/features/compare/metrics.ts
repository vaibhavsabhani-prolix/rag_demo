/** Numbers compared across collections, derived from each collection's SearchRun. */
import { formatBytes, formatMs } from '@/lib/format'
import { summarizePatent } from '@/lib/patent'
import type { SearchCollection } from '@/schemas/api'
import type { PhaseNumber } from '@/schemas/pipeline'
import type { SearchRun } from '@/store/searchSlice'

/** Phases that run once per collection (Phase 1 is shared). */
export const COLLECTION_PHASES: PhaseNumber[] = [2, 3, 4, 5, 6, 7]

/** How many top candidates the similarity chart plots. */
export const TOP_CANDIDATES = 20

export interface CollectionMetrics {
  name: string
  run: SearchRun
  phaseMs: Partial<Record<PhaseNumber, number>>
  /** Phases 2–7. */
  totalMs?: number
  /** Extra memory (bytes) each step needed at its peak, over what the process held when it began. */
  phaseMemory: Partial<Record<PhaseNumber, number>>
  /** Extra memory (bytes) at the peak of Phases 2–7. */
  peakMemory?: number
  /** The API process's whole resident memory (bytes) at that peak. */
  peakRss?: number
  // Collection info (static, from GET /api/collections - not from this query)
  /** Every patent in the collection, not just this query's matches. */
  collectionPatents?: number
  /** Every chunk/vector in the collection, not just this query's matches. */
  collectionChunks?: number
  /** Max tokens per chunk, parsed from the collection name (e.g. "512", "4096") when it's numeric. */
  chunkTokens?: number
  // Patents at each step
  candidates?: number
  afterFilter?: number
  verified?: number
  results?: number
  rejected?: number
  // Chunks at each step
  chunkHits?: number
  evidenceChunks?: number
  chunksReranked?: number
  // Phase 2 candidate retrieval latency breakdown
  embeddingMs?: number
  qdrantMs?: number
  // Scores
  finalScores: { patentId: string; score: number }[]
  topScore?: number
  meanScore?: number
  topSimilarities: { patentId: string; score: number }[]
  // Cross-encoder reranker scores (Phase 6/7), across final results
  topRerankerScore?: number
  meanRerankerScore?: number
  minRerankerScore?: number
}

export function collectionMetrics(name: string, run: SearchRun, info?: SearchCollection): CollectionMetrics {
  const { phases } = run
  const phaseMs: Partial<Record<PhaseNumber, number>> = {}
  const phaseMemory: Partial<Record<PhaseNumber, number>> = {}
  for (const id of COLLECTION_PHASES) {
    const entry = phases[id]
    if (entry) phaseMs[id] = entry.elapsedMs
    if (entry?.memory) phaseMemory[id] = entry.memory.peak_bytes - entry.memory.start_bytes
  }

  const results = phases[7]?.data.results ?? []
  // Excludes metadata-only results, which have no final_score to chart or rank by.
  const finalScores = results.flatMap((r) =>
    r.final_score === null ? [] : [{ patentId: r.patent_id, score: r.final_score }],
  )
  const scores = finalScores.map((s) => s.score)
  const rerankerScores = results.map((r) => r.best_reranker_score)
  const chunkTokens = Number(name)

  return {
    name,
    run,
    phaseMs,
    totalMs: run.totalMs,
    phaseMemory,
    peakMemory: run.memory && run.memory.peak_bytes - run.memory.start_bytes,
    peakRss: run.memory?.peak_bytes,
    collectionPatents: info?.patent_count,
    collectionChunks: info?.chunk_count,
    chunkTokens: Number.isFinite(chunkTokens) && chunkTokens > 0 ? chunkTokens : undefined,
    candidates: phases[2]?.data.candidates.length,
    afterFilter: phases[3]?.data.total_after,
    verified: phases[5]?.data.total_evaluated,
    results: phases[7] ? results.length : undefined,
    rejected: phases[7]?.data.rejected_count,
    chunkHits: phases[2]?.data.total_chunk_hits,
    evidenceChunks: phases[4]?.data.total_evidence_chunks,
    chunksReranked: phases[6]?.data.total_chunks_reranked,
    embeddingMs: phases[2]?.data.timings.embedding_ms,
    qdrantMs: phases[2]?.data.timings.qdrant_retrieval_ms,
    finalScores,
    topScore: scores.length ? Math.max(...scores) : undefined,
    meanScore: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : undefined,
    topSimilarities: (phases[2]?.data.candidates ?? [])
      .slice(0, TOP_CANDIDATES)
      .map((c) => ({ patentId: c.patent_id, score: c.retrieval_score })),
    topRerankerScore: rerankerScores.length ? Math.max(...rerankerScores) : undefined,
    meanRerankerScore: rerankerScores.length
      ? rerankerScores.reduce((a, b) => a + b, 0) / rerankerScores.length
      : undefined,
    minRerankerScore: rerankerScores.length ? Math.min(...rerankerScores) : undefined,
  }
}

export interface OverlapRow {
  patentId: string
  /** Title, or the assignee when the metadata has no title. */
  title: string
  /** Collection name -> rank (1-based) and score (null for a metadata-only query) in that collection's results. */
  found: Record<string, { rank: number; score: number | null }>
}

/** Every patent in any collection's final results, found-by-most first. */
export function resultOverlap(metrics: CollectionMetrics[]): OverlapRow[] {
  const rows = new Map<string, OverlapRow>()
  for (const m of metrics) {
    const results = m.run.phases[7]?.data.results ?? []
    results.forEach((r, i) => {
      let row = rows.get(r.patent_id)
      if (!row) {
        const { title, assignee } = summarizePatent(r.metadata)
        row = { patentId: r.patent_id, title: title || assignee, found: {} }
        rows.set(r.patent_id, row)
      }
      row.found[m.name] = { rank: i + 1, score: r.final_score }
    })
  }
  // Falls back to found-by-most only when every score is null (metadata-only results).
  const best = (row: OverlapRow) => {
    const scores = Object.values(row.found)
      .map((f) => f.score)
      .filter((s): s is number => s !== null)
    return scores.length ? Math.max(...scores) : -Infinity
  }
  return [...rows.values()].sort(
    (a, b) => Object.keys(b.found).length - Object.keys(a.found).length || best(b) - best(a),
  )
}

/** Candidate patents (Phase 2) that two collections have in common. */
export function sharedCandidates(a: CollectionMetrics, b: CollectionMetrics): number | undefined {
  const ca = a.run.phases[2]?.data.candidates
  const cb = b.run.phases[2]?.data.candidates
  if (!ca || !cb) return undefined
  const ids = new Set(ca.map((c) => c.patent_id))
  return cb.filter((c) => ids.has(c.patent_id)).length
}

export interface Finding {
  label: string
  /** Winning collection(s). */
  value: string
  detail: string
}

/** The collection(s) with the best value, when at least two collections have one. */
function best(metrics: CollectionMetrics[], get: (m: CollectionMetrics) => number | undefined, higher: boolean) {
  const values = metrics.flatMap((m) => {
    const v = get(m)
    return v === undefined ? [] : [{ name: m.name, value: v }]
  })
  if (values.length < 2) return undefined
  const target = (higher ? Math.max : Math.min)(...values.map((v) => v.value))
  const worst = (higher ? Math.min : Math.max)(...values.map((v) => v.value))
  return {
    names: values.filter((v) => v.value === target).map((v) => v.name),
    value: target,
    worst: values.find((v) => v.value === worst)!,
    values,
  }
}

/** Plain-language answers to "which collection did best?", from finished runs only. */
export function keyFindings(metrics: CollectionMetrics[]): Finding[] {
  const done = metrics.filter((m) => m.run.status === 'success')
  const findings: Finding[] = []

  const fastest = best(done, (m) => m.totalMs, false)
  if (fastest) {
    const faster =
      fastest.worst.value > fastest.value
        ? ` · ${(fastest.worst.value / fastest.value).toFixed(1)}× faster than ${fastest.worst.name}`
        : ''
    findings.push({
      label: 'Fastest',
      value: fastest.names.join(' & '),
      detail: `${formatMs(fastest.value)} for steps 2–7${faster}`,
    })
  }

  const lightest = best(done, (m) => m.peakMemory, false)
  if (lightest) {
    const less =
      lightest.worst.value > lightest.value
        ? ` · ${formatBytes(lightest.worst.value - lightest.value)} less than ${lightest.worst.name}`
        : ''
    findings.push({
      label: 'Least memory',
      value: lightest.names.join(' & '),
      detail: `+${formatBytes(lightest.value)} at the peak of steps 2–7${less}`,
    })
  }

  const most = best(done, (m) => m.results, true)
  if (most) {
    const others = most.values
      .filter((v) => !most.names.includes(v.name))
      .map((v) => `${v.name}: ${v.value}`)
      .join(', ')
    findings.push({
      label: 'Most results',
      value: most.names.join(' & '),
      detail: `${most.value} qualifying patents${others ? ` (${others})` : ''}`,
    })
  }

  const top = best(done, (m) => m.topScore, true)
  if (top) {
    findings.push({
      label: 'Best top result',
      value: top.names.join(' & '),
      detail: `score ${top.value.toFixed(2)} / 10`,
    })
  }

  const mean = best(done, (m) => m.meanScore, true)
  if (mean) {
    findings.push({
      label: 'Best average score',
      value: mean.names.join(' & '),
      detail: `${mean.value.toFixed(2)} / 10 across its results`,
    })
  }

  if (done.length >= 2) {
    const rows = resultOverlap(done)
    const shared = rows.filter((r) => Object.keys(r.found).length === done.length).length
    findings.push({
      label: 'Found by every collection',
      value: `${shared} patent${shared === 1 ? '' : 's'}`,
      detail: `out of ${rows.length} different qualifying patents`,
    })
  }

  return findings
}

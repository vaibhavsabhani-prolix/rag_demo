import { useMemo, useState } from 'react'
import { HighlightedText } from '@/components/patent/EvidenceChunkCard'
import { Badge, ChevronIcon, ExternalIcon } from '@/components/ui'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type {
  KeyFeatureItem,
  KeyFeatureSearchResponse,
  FeatureSearchResult,
  FeaturePatentResult,
  FeatureChunkResult,
} from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'

interface FeatureWiseMatrixProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse }>
  series: Series[]
}

interface FeatureStatsPerCollection {
  collection: string
  patentsCount: number
  bestScore: number
  topPatentId: string | null
  patentIds: Set<string>
  featureResult?: FeatureSearchResult
}

interface FeatureMatrixRow {
  feature_id: number
  feature: string
  byCollection: Record<string, FeatureStatsPerCollection>
  sharedPatentIds: string[]
  totalUniquePatentIds: string[]
  agreementLevel: 'high' | 'medium' | 'low' | 'none'
}

interface CommonPatentComparison {
  patent_id: string
  title?: string | null
  byCollection: Record<string, { rank: number; score: number; chunksCount: number }>
  matchedCount: number
  avgRank: number
  bestScoreOverall: number
}

function extractFeaturePatents(featRes?: FeatureSearchResult): FeaturePatentResult[] {
  if (!featRes) return []
  if (featRes.patents && featRes.patents.length > 0) {
    return featRes.patents
  }
  // Group chunks by patent_id fallback
  const map = new Map<string, FeaturePatentResult>()
  for (const chunk of featRes.chunks || []) {
    let p = map.get(chunk.patent_id)
    if (!p) {
      p = {
        patent_id: chunk.patent_id,
        title: chunk.title ?? null,
        score: chunk.score,
        best_reranker_score: chunk.reranker_score ?? null,
        best_retrieval_score: chunk.retrieval_score ?? null,
        metadata: chunk.metadata || {},
        chunks: [],
      }
      map.set(chunk.patent_id, p)
    }
    p.chunks.push(chunk)
    p.score = Math.max(p.score, chunk.score)
    if (chunk.reranker_score != null) {
      p.best_reranker_score = Math.max(p.best_reranker_score ?? 0, chunk.reranker_score)
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => (b.best_reranker_score ?? b.score) - (a.best_reranker_score ?? a.score),
  )
}

function scoreTone(score: number): 'brand' | 'success' | 'warning' | 'neutral' {
  if (score >= 0.8) return 'success'
  if (score >= 0.6) return 'brand'
  if (score >= 0.4) return 'warning'
  return 'neutral'
}

export function FeatureWiseMatrix({ features, runs, series }: FeatureWiseMatrixProps) {
  const { data: config } = usePipelineConfig()
  const [selectedFeatureId, setSelectedFeatureId] = useState<number>(features[0]?.feature_id ?? 1)
  const [expandedChunkPatentKey, setExpandedChunkPatentKey] = useState<string | null>(null)
  const [commonViewMode, setCommonViewMode] = useState<'table' | 'pills'>('table')

  const collectionNames = series.map((s) => s.name)

  // Build matrix data comparing every feature across all collections
  const matrixRows = useMemo<FeatureMatrixRow[]>(() => {
    return features.map((feat) => {
      const byCollection: Record<string, FeatureStatsPerCollection> = {}
      const allSets: Set<string>[] = []
      const combinedUnique = new Set<string>()

      for (const collName of collectionNames) {
        const resp = runs[collName]?.response
        const featRes = resp?.results?.find((r) => r.feature_id === feat.feature_id)
        const patents = extractFeaturePatents(featRes)

        let bestScore = 0
        let topPatentId: string | null = null
        const patentIds = new Set<string>()

        for (const p of patents) {
          patentIds.add(p.patent_id)
          combinedUnique.add(p.patent_id)
          const sc = p.best_reranker_score ?? p.score
          if (sc > bestScore) {
            bestScore = sc
            topPatentId = p.patent_id
          }
        }

        if (patentIds.size > 0) {
          allSets.push(patentIds)
        }

        byCollection[collName] = {
          collection: collName,
          patentsCount: patentIds.size,
          bestScore,
          topPatentId,
          patentIds,
          featureResult: featRes,
        }
      }

      // Shared across ALL collections (must be present in every compared collection for this feature)
      let sharedPatentIds: string[] = []
      if (collectionNames.length > 1) {
        const allCollectionSets = collectionNames.map(
          (name) => byCollection[name]?.patentIds ?? new Set<string>(),
        )
        if (allCollectionSets.every((set) => set.size > 0)) {
          const [firstSet, ...restSets] = allCollectionSets
          sharedPatentIds = Array.from(firstSet).filter((pid) =>
            restSets.every((set) => set.has(pid)),
          )
        }
      }

      const totalUnique = Array.from(combinedUnique)
      let agreementLevel: FeatureMatrixRow['agreementLevel'] = 'none'
      if (sharedPatentIds.length >= 3) {
        agreementLevel = 'high'
      } else if (sharedPatentIds.length > 0) {
        agreementLevel = 'medium'
      } else if (totalUnique.length > 0) {
        agreementLevel = 'low'
      }

      return {
        feature_id: feat.feature_id,
        feature: feat.feature,
        byCollection,
        sharedPatentIds,
        totalUniquePatentIds: totalUnique,
        agreementLevel,
      }
    })
  }, [features, runs, collectionNames])

  const selectedRow = matrixRows.find((r) => r.feature_id === selectedFeatureId) ?? matrixRows[0]

  // Map out common patents with their rank number and score across each collection
  const commonPatentsList = useMemo<CommonPatentComparison[]>(() => {
    if (!selectedRow || selectedRow.sharedPatentIds.length === 0) return []

    const collectionPatentMap: Record<
      string,
      Map<string, { rank: number; score: number; chunksCount: number; title?: string | null }>
    > = {}

    for (const s of series) {
      const patents = extractFeaturePatents(selectedRow.byCollection[s.name]?.featureResult)
      const map = new Map<string, { rank: number; score: number; chunksCount: number; title?: string | null }>()
      patents.forEach((p, idx) => {
        map.set(p.patent_id, {
          rank: idx + 1,
          score: p.best_reranker_score ?? p.score,
          chunksCount: p.chunks?.length || 0,
          title: p.title,
        })
      })
      collectionPatentMap[s.name] = map
    }

    return selectedRow.sharedPatentIds
      .map((pid) => {
        const byCol: Record<string, { rank: number; score: number; chunksCount: number }> = {}
        let rankSum = 0
        let rankCount = 0
        let bestScore = 0
        let title: string | null | undefined = null

        for (const s of series) {
          const info = collectionPatentMap[s.name]?.get(pid)
          if (info) {
            byCol[s.name] = { rank: info.rank, score: info.score, chunksCount: info.chunksCount }
            rankSum += info.rank
            rankCount += 1
            if (info.score > bestScore) bestScore = info.score
            if (!title && info.title) title = info.title
          }
        }

        const avgRank = rankCount > 0 ? rankSum / rankCount : 999

        return {
          patent_id: pid,
          title,
          byCollection: byCol,
          matchedCount: rankCount,
          avgRank,
          bestScoreOverall: bestScore,
        }
      })
      .sort((a, b) => a.avgRank - b.avgRank || b.bestScoreOverall - a.bestScoreOverall)
  }, [selectedRow, series])

  const toggleChunkExpansion = (key: string) => {
    setExpandedChunkPatentKey((prev) => (prev === key ? null : key))
  }

  return (
    <div className="space-y-6">
      {/* Side-by-Side Deep Dive Inspector for Selected Feature */}
      {selectedRow && (
        <div className="rounded-xl border border-indigo-200 dark:border-indigo-900/60 bg-surface shadow-xs overflow-hidden">
          {/* Header of Inspector */}
          <div className="border-b border-indigo-100 dark:border-indigo-900/50 bg-indigo-50/60 dark:bg-indigo-950/20 p-4 sm:p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-start gap-3">
                <Badge tone="brand" className="font-mono text-xs font-bold shrink-0 mt-0.5">
                  Feature {selectedRow.feature_id} Deep Dive
                </Badge>
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white leading-snug">
                    {selectedRow.feature}
                  </h3>
                  <p className="mt-1 text-xs text-slate-600 dark:text-white/80">
                    Comparing matched patents, scores, and text chunks side by side across collections.
                  </p>
                </div>
              </div>

              {/* Feature Switcher Quick Pills */}
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-2xs font-medium text-slate-500 dark:text-white/70 mr-1">Switch:</span>
                {features.map((f) => (
                  <button
                    key={f.feature_id}
                    type="button"
                    onClick={() => setSelectedFeatureId(f.feature_id)}
                    className={`size-6 rounded-md text-xs font-mono font-bold transition-colors ${
                      f.feature_id === selectedFeatureId
                        ? 'bg-indigo-600 text-white shadow-2xs'
                        : 'bg-surface border border-slate-200 dark:border-white/15 text-slate-600 dark:text-white hover:bg-slate-100 dark:hover:bg-white/10'
                    }`}
                  >
                    {f.feature_id}
                  </button>
                ))}
              </div>
            </div>

            {/* Common Patents Across Collections Comparison Table */}
            {commonPatentsList.length > 0 && (
              <div className="mt-4 rounded-xl border border-emerald-500/30 dark:border-emerald-600/30 bg-surface shadow-xs overflow-hidden">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-white/10 bg-slate-50/80 dark:bg-white/5 px-4 py-2.5">
                  <div className="flex items-center gap-2">
                    <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-white text-2xs font-bold shadow-xs">
                      ✓
                    </span>
                    <div>
                      <span className="font-semibold text-xs text-slate-900 dark:text-white">
                        Common Patents in All {collectionNames.length} Collections ({commonPatentsList.length})
                      </span>
                      <span className="text-2xs text-slate-500 dark:text-white/60 ml-2 hidden sm:inline">
                        • Shows the exact position (#) and rerank score in each collection
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 text-2xs font-medium">
                    <button
                      type="button"
                      onClick={() => setCommonViewMode('table')}
                      className={`px-2.5 py-1 rounded transition-colors ${
                        commonViewMode === 'table'
                          ? 'bg-indigo-600 text-white shadow-2xs font-semibold'
                          : 'bg-surface border border-slate-200 dark:border-white/10 text-slate-600 dark:text-white hover:bg-slate-100 dark:hover:bg-white/10'
                      }`}
                    >
                      Rank Table
                    </button>
                    <button
                      type="button"
                      onClick={() => setCommonViewMode('pills')}
                      className={`px-2.5 py-1 rounded transition-colors ${
                        commonViewMode === 'pills'
                          ? 'bg-indigo-600 text-white shadow-2xs font-semibold'
                          : 'bg-surface border border-slate-200 dark:border-white/10 text-slate-600 dark:text-white hover:bg-slate-100 dark:hover:bg-white/10'
                      }`}
                    >
                      Pills View
                    </button>
                  </div>
                </div>

                {commonViewMode === 'table' ? (
                  <div className="overflow-x-auto max-h-72 overflow-y-auto">
                    <table className="min-w-full divide-y divide-slate-200 dark:divide-white/10 text-xs">
                      <thead className="bg-slate-50 dark:bg-surface sticky top-0 z-10">
                        <tr>
                          <th className="px-3.5 py-2 text-left font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                            Common Patent
                          </th>
                          {series.map((s) => (
                            <th
                              key={s.name}
                              className="px-3.5 py-2 text-left font-semibold text-slate-700 dark:text-white uppercase text-2xs"
                            >
                              <div className="flex items-center gap-1.5">
                                <span className="size-2 rounded-sm" style={{ backgroundColor: s.color }} />
                                <span>{s.name} Position & Score</span>
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-white/10 bg-surface">
                        {commonPatentsList.map((cp) => {
                          const url = patentUrl(config?.patent_view_url_template, cp.patent_id)
                          return (
                            <tr
                              key={cp.patent_id}
                              className="hover:bg-slate-50/70 dark:hover:bg-white/5 transition-colors"
                            >
                              <td className="px-3.5 py-2.5 align-middle">
                                <div className="flex flex-col">
                                  {url ? (
                                    <a
                                      href={url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="font-mono text-xs font-bold text-indigo-600 hover:underline dark:text-white inline-flex items-center gap-1"
                                      title="View Patent Document"
                                    >
                                      <span>{cp.patent_id}</span>
                                      <ExternalIcon className="size-2.5 opacity-60" />
                                    </a>
                                  ) : (
                                    <span className="font-mono text-xs font-bold text-slate-900 dark:text-white">
                                      {cp.patent_id}
                                    </span>
                                  )}
                                  {cp.title && (
                                    <span
                                      className="text-2xs text-slate-500 dark:text-white/70 truncate max-w-[240px]"
                                      title={cp.title}
                                    >
                                      {cp.title}
                                    </span>
                                  )}
                                </div>
                              </td>

                              {series.map((s) => {
                                const col = cp.byCollection[s.name]
                                if (!col) {
                                  return (
                                    <td
                                      key={s.name}
                                      className="px-3.5 py-2.5 align-middle text-slate-400 italic text-2xs"
                                    >
                                      Not in top
                                    </td>
                                  )
                                }
                                return (
                                  <td key={s.name} className="px-3.5 py-2.5 align-middle">
                                    <div className="flex items-center gap-2">
                                      <span
                                        className={`inline-flex items-center justify-center px-2 py-0.5 rounded font-mono text-2xs font-bold border ${
                                          col.rank === 1
                                            ? 'bg-amber-100 dark:bg-amber-500/20 text-amber-900 dark:text-amber-200 border-amber-300 dark:border-amber-500/40'
                                            : 'bg-slate-100 dark:bg-white/10 text-slate-800 dark:text-white border-slate-200 dark:border-white/15'
                                        }`}
                                      >
                                        #{col.rank}
                                      </span>
                                      <Badge
                                        tone={scoreTone(col.score)}
                                        className="font-mono text-2xs font-bold dark:text-white"
                                      >
                                        {col.score.toFixed(3)}
                                      </Badge>
                                      <span className="text-2xs text-slate-400 dark:text-white/50">
                                        ({col.chunksCount} chk)
                                      </span>
                                    </div>
                                  </td>
                                )
                              })}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-3 flex flex-wrap gap-1.5">
                    {selectedRow.sharedPatentIds.map((pid) => (
                      <span
                        key={pid}
                        className="font-mono text-2xs font-bold text-emerald-800 dark:text-white bg-emerald-50 dark:bg-white/10 px-2 py-0.5 rounded border border-emerald-200 dark:border-emerald-500/40"
                      >
                        {pid}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Side-by-Side Multi-Column Comparison */}
          <div className="p-4 sm:p-5">
            <div
              className={`grid grid-cols-1 gap-5 ${
                collectionNames.length === 2
                  ? 'md:grid-cols-2'
                  : collectionNames.length >= 3
                  ? 'md:grid-cols-3'
                  : ''
              }`}
            >
              {series.map((s) => {
                const colData = selectedRow.byCollection[s.name]
                const patents = extractFeaturePatents(colData?.featureResult)

                return (
                  <div
                    key={s.name}
                    className="flex flex-col rounded-xl border border-slate-200 dark:border-white/10 bg-surface overflow-hidden shadow-2xs"
                  >
                    {/* Column Header */}
                    <div className="flex items-center justify-between gap-2 border-b border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-surface px-3.5 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="size-3 rounded-xs shrink-0" style={{ backgroundColor: s.color }} />
                        <span className="font-bold text-sm text-slate-900 dark:text-white">{s.name}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Badge tone="neutral" className="text-2xs font-semibold">
                          {patents.length} {patents.length === 1 ? 'patent' : 'patents'}
                        </Badge>
                        {colData?.bestScore ? (
                          <Badge tone={scoreTone(colData.bestScore)} className="font-mono text-2xs font-bold">
                            Top: {colData.bestScore.toFixed(3)}
                          </Badge>
                        ) : null}
                      </div>
                    </div>

                    {/* Patent List for this collection */}
                    <div className="p-3 space-y-2.5 max-h-[38rem] overflow-y-auto flex-1">
                      {patents.length === 0 ? (
                        <p className="p-4 text-center text-xs text-slate-400 dark:text-white/40 italic">
                          No candidate matched this feature in collection “{s.name}”.
                        </p>
                      ) : (
                        patents.map((patent, rankIdx) => {
                          const isShared = selectedRow.sharedPatentIds.includes(patent.patent_id)
                          const patentKey = `${s.name}-${patent.patent_id}`
                          const isChunksExpanded = expandedChunkPatentKey === patentKey
                          const url = patentUrl(config?.patent_view_url_template, patent.patent_id)

                          return (
                            <div
                              key={patent.patent_id}
                              id={`patent-card-${patent.patent_id}`}
                              className={`rounded-lg border bg-surface transition-all ${
                                isShared
                                  ? 'border-emerald-300/80 dark:border-emerald-600/60 shadow-2xs'
                                  : 'border-slate-200 dark:border-white/10'
                              }`}
                            >
                              {/* Patent Summary Card Header */}
                              <div className="p-3">
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex items-center gap-1.5 min-w-0">
                                    <span className="flex size-5 items-center justify-center rounded bg-slate-100 dark:bg-white/10 border border-slate-200 dark:border-white/10 font-mono text-2xs font-bold text-slate-700 dark:text-white shrink-0">
                                      #{rankIdx + 1}
                                    </span>
                                    {url ? (
                                      <a
                                        href={url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="font-mono text-xs font-bold text-indigo-600 hover:underline dark:text-white inline-flex items-center gap-1 truncate"
                                        title="View Patent Document"
                                      >
                                        <span>{patent.patent_id}</span>
                                        <ExternalIcon className="size-2.5 opacity-60" />
                                      </a>
                                    ) : (
                                      <span className="font-mono text-xs font-bold text-indigo-700 dark:text-white truncate">
                                        {patent.patent_id}
                                      </span>
                                    )}
                                  </div>

                                  <div className="flex items-center gap-1 shrink-0">
                                    {isShared && (
                                      <Badge tone="success" className="text-2xs font-semibold px-1.5 py-0 dark:text-white">
                                        Shared
                                      </Badge>
                                    )}
                                    <Badge
                                      tone={scoreTone(patent.best_reranker_score ?? patent.score)}
                                      className="font-mono text-2xs font-bold dark:text-white"
                                    >
                                      {(patent.best_reranker_score ?? patent.score).toFixed(3)}
                                    </Badge>
                                  </div>
                                </div>

                                {patent.title && (
                                  <p className="mt-1 text-2xs text-slate-600 dark:text-white/80 line-clamp-2" title={patent.title}>
                                    {patent.title}
                                  </p>
                                )}

                                {/* Card Footer with Chunk Toggle Button */}
                                <div className="mt-2.5 flex items-center justify-between border-t border-slate-200 dark:border-white/10 pt-2 text-2xs text-slate-500 dark:text-white/70">
                                  <span>
                                    {patent.chunks?.length || 0} chunk{patent.chunks?.length === 1 ? '' : 's'}
                                    {patent.best_retrieval_score != null && (
                                      <span className="text-slate-400 dark:text-white/50 font-mono ml-1.5">
                                        (Vec: {patent.best_retrieval_score.toFixed(2)})
                                      </span>
                                    )}
                                  </span>

                                  {patent.chunks && patent.chunks.length > 0 && (
                                    <button
                                      type="button"
                                      onClick={() => toggleChunkExpansion(patentKey)}
                                      className="inline-flex items-center gap-1 text-indigo-600 font-medium hover:text-indigo-800 dark:text-white dark:hover:text-white/80"
                                    >
                                      <span>{isChunksExpanded ? 'Hide chunks' : 'View chunks'}</span>
                                      <ChevronIcon
                                        className={`size-3 transition-transform ${
                                          isChunksExpanded ? 'rotate-90' : ''
                                        }`}
                                      />
                                    </button>
                                  )}
                                </div>
                              </div>

                              {/* Expanded Chunks Passage Viewer */}
                              {isChunksExpanded && patent.chunks && patent.chunks.length > 0 && (
                                <div className="border-t border-slate-200 dark:border-white/10 bg-slate-50/70 dark:bg-surface p-2.5 space-y-2 text-2xs">
                                  {patent.chunks.map((ch: FeatureChunkResult, chIdx: number) => (
                                    <div
                                      key={ch.chunk_id || chIdx}
                                      className="rounded border border-slate-200 dark:border-white/10 bg-surface p-2 shadow-2xs space-y-1.5"
                                    >
                                      <div className="flex items-center justify-between text-2xs text-slate-400 dark:text-white/60 border-b border-slate-200 dark:border-white/10 pb-1">
                                        <span className="font-semibold text-slate-700 dark:text-white truncate max-w-[140px]">
                                          {ch.section || `Chunk ${ch.chunk_id}`}
                                        </span>
                                        <div className="flex items-center gap-1 font-mono">
                                          {ch.reranker_score != null && (
                                            <span className="font-bold text-slate-900 dark:text-white">
                                              Rerank: {ch.reranker_score.toFixed(3)}
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                      <div className="text-slate-800 dark:text-white leading-relaxed max-h-32 overflow-y-auto text-2xs">
                                        <HighlightedText
                                          text={ch.text}
                                          limit={1000}
                                          highlight={ch.highlight}
                                        />
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )
                        })
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

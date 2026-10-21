import { useMemo, useState } from 'react'
import { Badge, CheckIcon, ChevronIcon, ExternalIcon, TextField } from '@/components/ui'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type {
  KeyFeatureItem,
  KeyFeatureSearchResponse,
  FeatureSearchResult,
  FeatureChunkResult,
} from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'

interface PatentResultsMatrixProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse }>
  series: Series[]
}

interface CollectionResultDetail {
  matched: boolean
  bestScore: number
  matchedFeatureIds: number[]
  chunks: FeatureChunkResult[]
}

interface PatentMatrixRow {
  patent_id: string
  title?: string | null
  bestScoreOverall: number
  // Under the active feature filter:
  matchedCollections: string[]
  matchedFeatureIds: number[]
  byCollection: Record<string, CollectionResultDetail>
  totalChunks: number
}

function scoreTone(score: number): 'brand' | 'success' | 'warning' | 'neutral' {
  if (score >= 0.8) return 'success'
  if (score >= 0.6) return 'brand'
  if (score >= 0.4) return 'warning'
  return 'neutral'
}

/**
 * Extracts winning result patents from a FeatureSearchResult.
 * Checks `featRes.patents` (final winners), then `featRes.final_patents`,
 * and falls back to grouping `featRes.chunks`.
 */
function extractWinningPatents(featRes?: FeatureSearchResult): Map<
  string,
  { title?: string | null; score: number; chunks: FeatureChunkResult[] }
> {
  const result = new Map<string, { title?: string | null; score: number; chunks: FeatureChunkResult[] }>()
  if (!featRes) return result

  if (featRes.patents && featRes.patents.length > 0) {
    for (const p of featRes.patents) {
      const existing = result.get(p.patent_id)
      const sc = p.best_reranker_score ?? p.score
      const chunks = (p.chunks || []).map((ch) => ({
        ...ch,
        feature_id: ch.feature_id ?? featRes.feature_id,
      }))
      if (!existing) {
        result.set(p.patent_id, {
          title: p.title,
          score: sc,
          chunks,
        })
      } else {
        existing.score = Math.max(existing.score, sc)
        for (const ch of chunks) {
          if (!existing.chunks.some((c) => c.chunk_id === ch.chunk_id)) {
            existing.chunks.push(ch)
          }
        }
      }
    }
    return result
  }

  // Fallback: group chunks
  for (const ch of featRes.chunks || []) {
    const sc = ch.reranker_score ?? ch.score
    const existing = result.get(ch.patent_id)
    const taggedChunk: FeatureChunkResult = {
      ...ch,
      feature_id: ch.feature_id ?? featRes.feature_id,
    }
    if (!existing) {
      result.set(ch.patent_id, {
        title: ch.title,
        score: sc,
        chunks: [taggedChunk],
      })
    } else {
      existing.score = Math.max(existing.score, sc)
      if (!existing.chunks.some((c) => c.chunk_id === ch.chunk_id)) {
        existing.chunks.push(taggedChunk)
      }
    }
  }

  return result
}

export function PatentResultsMatrix({ features, runs, series }: PatentResultsMatrixProps) {
  const { data: config } = usePipelineConfig()
  const [selectedFeatureId, setSelectedFeatureId] = useState<'all' | number>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [consensusFilter, setConsensusFilter] = useState<'all' | 'all_collections' | 'multi' | 'exclusive'>('all')
  const [sortBy, setSortBy] = useState<'collections' | 'score' | 'features' | 'id'>('collections')
  const [expandedPatentId, setExpandedPatentId] = useState<string | null>(null)

  // Pagination state
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(25)

  const collectionNames = useMemo(() => series.map((s) => s.name), [series])

  // Build matrix data based on the selected feature
  const matrixData = useMemo(() => {
    const patentMap = new Map<string, PatentMatrixRow>()

    // Helper to get or init row
    const getRow = (patentId: string, title?: string | null): PatentMatrixRow => {
      let row = patentMap.get(patentId)
      if (!row) {
        const byCollection: Record<string, CollectionResultDetail> = {}
        for (const c of collectionNames) {
          byCollection[c] = {
            matched: false,
            bestScore: 0,
            matchedFeatureIds: [],
            chunks: [],
          }
        }
        row = {
          patent_id: patentId,
          title: title ?? null,
          bestScoreOverall: 0,
          matchedCollections: [],
          matchedFeatureIds: [],
          byCollection,
          totalChunks: 0,
        }
        patentMap.set(patentId, row)
      } else if (!row.title && title) {
        row.title = title
      }
      return row
    }

    // Iterate through all collections
    for (const colName of collectionNames) {
      const resp = runs[colName]?.response
      if (!resp) continue

      const activeResults =
        selectedFeatureId === 'all'
          ? resp.results || []
          : (resp.results || []).filter((r) => r.feature_id === selectedFeatureId)

      for (const featRes of activeResults) {
        const featId = featRes.feature_id
        const winningMap = extractWinningPatents(featRes)

        for (const [patentId, info] of winningMap.entries()) {
          const row = getRow(patentId, info.title)

          // Mark collection as matched for this patent
          const colDetail = row.byCollection[colName]
          colDetail.matched = true
          colDetail.bestScore = Math.max(colDetail.bestScore, info.score)
          if (!colDetail.matchedFeatureIds.includes(featId)) {
            colDetail.matchedFeatureIds.push(featId)
          }

          // Append unique chunks
          for (const ch of info.chunks) {
            if (!colDetail.chunks.some((c) => c.chunk_id === ch.chunk_id)) {
              colDetail.chunks.push(ch)
            }
          }

          if (!row.matchedCollections.includes(colName)) {
            row.matchedCollections.push(colName)
          }
          if (!row.matchedFeatureIds.includes(featId)) {
            row.matchedFeatureIds.push(featId)
          }

          row.bestScoreOverall = Math.max(row.bestScoreOverall, info.score)
        }
      }
    }

    // Calculate total chunks for each patent across all collections
    for (const row of patentMap.values()) {
      let chunksCount = 0
      for (const c of collectionNames) {
        chunksCount += row.byCollection[c].chunks.length
      }
      row.totalChunks = chunksCount
    }

    // Convert map to list
    const rows = Array.from(patentMap.values())

    // Summary counts per collection under current feature filter
    const countPerCollection: Record<string, number> = {}
    for (const c of collectionNames) {
      countPerCollection[c] = rows.filter((r) => r.byCollection[c].matched).length
    }

    return { rows, countPerCollection }
  }, [runs, collectionNames, selectedFeatureId])

  // Filtered rows
  const filteredRows = useMemo(() => {
    let list = matrixData.rows

    // Search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim()
      list = list.filter(
        (r) =>
          r.patent_id.toLowerCase().includes(q) ||
          (r.title && r.title.toLowerCase().includes(q))
      )
    }

    // Consensus filter
    if (consensusFilter === 'all_collections') {
      list = list.filter((r) => r.matchedCollections.length === collectionNames.length)
    } else if (consensusFilter === 'multi') {
      list = list.filter((r) => r.matchedCollections.length >= 2)
    } else if (consensusFilter === 'exclusive') {
      list = list.filter((r) => r.matchedCollections.length === 1)
    }

    // Sorting
    list = [...list].sort((a, b) => {
      if (sortBy === 'collections') {
        const diff = b.matchedCollections.length - a.matchedCollections.length
        if (diff !== 0) return diff
        return b.bestScoreOverall - a.bestScoreOverall
      }
      if (sortBy === 'score') {
        return b.bestScoreOverall - a.bestScoreOverall
      }
      if (sortBy === 'features') {
        const diff = b.matchedFeatureIds.length - a.matchedFeatureIds.length
        if (diff !== 0) return diff
        return b.bestScoreOverall - a.bestScoreOverall
      }
      if (sortBy === 'id') {
        return a.patent_id.localeCompare(b.patent_id)
      }
      return 0
    })

    return list
  }, [matrixData.rows, searchQuery, consensusFilter, sortBy, collectionNames])

  // Pagination calculation
  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1
  const safePage = Math.min(page, totalPages - 1)
  const pagedRows = useMemo(() => {
    const start = safePage * pageSize
    return filteredRows.slice(start, start + pageSize)
  }, [filteredRows, safePage, pageSize])

  const handlePageChange = (newPage: number) => {
    setPage(Math.max(0, Math.min(totalPages - 1, newPage)))
  }

  // Active feature item if one is selected
  const activeFeatureObj = useMemo(() => {
    if (selectedFeatureId === 'all') return null
    return features.find((f) => f.feature_id === selectedFeatureId) ?? null
  }, [features, selectedFeatureId])

  // Global aggregate stats
  const totalPatentsCount = matrixData.rows.length
  const fullConsensusCount = matrixData.rows.filter(
    (r) => r.matchedCollections.length === collectionNames.length
  ).length
  const multiCollectionCount = matrixData.rows.filter(
    (r) => r.matchedCollections.length >= 2
  ).length
  const singleCollectionCount = matrixData.rows.filter(
    (r) => r.matchedCollections.length === 1
  ).length

  return (
    <div className="space-y-4">
      {/* Top Header & Feature Filter Section */}
      <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs p-4 sm:p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wide flex items-center gap-2">
              <span>Patent Results Matrix</span>
              <Badge tone="brand" className="text-2xs font-mono font-bold">
                {totalPatentsCount} Unique Patents
              </Badge>
            </h3>
            <p className="text-xs text-slate-500 dark:text-white/70 mt-0.5">
              Tracks which candidate patents came as final results in each collection. Columns show each collection with a tick mark (✓) when the patent was returned.
            </p>
          </div>

          {/* Quick Metrics Badges */}
          <div className="flex flex-wrap items-center gap-2 text-2xs font-medium">
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/20 bg-emerald-50/60 dark:bg-emerald-500/10 px-2.5 py-1 text-emerald-800 dark:text-emerald-300">
              <span className="size-1.5 rounded-full bg-emerald-500" />
              Full Consensus: <strong className="font-mono font-bold">{fullConsensusCount}</strong>
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-500/20 bg-indigo-50/60 dark:bg-indigo-500/10 px-2.5 py-1 text-indigo-800 dark:text-indigo-300">
              <span className="size-1.5 rounded-full bg-indigo-500" />
              2+ Collections: <strong className="font-mono font-bold">{multiCollectionCount}</strong>
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 dark:border-white/10 bg-slate-50/60 dark:bg-white/5 px-2.5 py-1 text-slate-700 dark:text-white/80">
              Single Collection: <strong className="font-mono font-bold">{singleCollectionCount}</strong>
            </span>
          </div>
        </div>

        {/* Key Feature Filter Buttons */}
        <div className="pt-2 border-t border-slate-100 dark:border-white/5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-2xs font-bold uppercase tracking-wider text-slate-500 dark:text-white/60 mr-1">
              Filter by Key Feature:
            </span>

            <button
              type="button"
              onClick={() => {
                setSelectedFeatureId('all')
                setPage(0)
              }}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                selectedFeatureId === 'all'
                  ? 'bg-indigo-600 text-white shadow-xs font-bold ring-2 ring-indigo-500/40'
                  : 'bg-surface text-slate-700 dark:text-white/80 border border-slate-200 dark:border-white/10 hover:bg-slate-100 dark:hover:bg-white/10'
              }`}
            >
              All Features ({features.length})
            </button>

            {features.map((feat) => {
              const isSelected = selectedFeatureId === feat.feature_id
              return (
                <button
                  key={feat.feature_id}
                  type="button"
                  onClick={() => {
                    setSelectedFeatureId(feat.feature_id)
                    setPage(0)
                  }}
                  className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all flex items-center gap-1 ${
                    isSelected
                      ? 'bg-indigo-600 text-white shadow-xs font-bold ring-2 ring-indigo-500/40'
                      : 'bg-surface text-slate-700 dark:text-white/80 border border-slate-200 dark:border-white/10 hover:bg-slate-100 dark:hover:bg-white/10'
                  }`}
                  title={`Feature ${feat.feature_id}: ${feat.feature}`}
                >
                  <span>F{feat.feature_id}</span>
                </button>
              )
            })}
          </div>

          {/* Active Feature Banner */}
          {activeFeatureObj && (
            <div className="mt-2.5 flex items-start gap-2 rounded-lg border border-indigo-200 bg-indigo-50/70 p-2.5 text-xs text-indigo-950 dark:border-indigo-800/40 dark:bg-indigo-950/20 dark:text-indigo-200">
              <span className="font-mono font-bold shrink-0 text-indigo-600 dark:text-indigo-400">
                F{activeFeatureObj.feature_id}:
              </span>
              <span className="line-clamp-2">{activeFeatureObj.feature}</span>
            </div>
          )}
        </div>

        {/* Secondary Filter & Search Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100 dark:border-white/5 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-2xs font-semibold text-slate-500 dark:text-white/60">Consensus:</span>
            <div className="inline-flex rounded-lg border border-slate-200 dark:border-white/15 p-0.5 bg-slate-50 dark:bg-white/5">
              <button
                type="button"
                onClick={() => {
                  setConsensusFilter('all')
                  setPage(0)
                }}
                className={`px-2 py-0.5 rounded text-2xs font-medium transition-all ${
                  consensusFilter === 'all'
                    ? 'bg-surface text-slate-900 dark:text-white shadow-2xs font-bold'
                    : 'text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                All ({matrixData.rows.length})
              </button>
              <button
                type="button"
                onClick={() => {
                  setConsensusFilter('all_collections')
                  setPage(0)
                }}
                className={`px-2 py-0.5 rounded text-2xs font-medium transition-all ${
                  consensusFilter === 'all_collections'
                    ? 'bg-surface text-slate-900 dark:text-white shadow-2xs font-bold'
                    : 'text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                All Collections ({fullConsensusCount})
              </button>
              <button
                type="button"
                onClick={() => {
                  setConsensusFilter('multi')
                  setPage(0)
                }}
                className={`px-2 py-0.5 rounded text-2xs font-medium transition-all ${
                  consensusFilter === 'multi'
                    ? 'bg-surface text-slate-900 dark:text-white shadow-2xs font-bold'
                    : 'text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                2+ Collections ({multiCollectionCount})
              </button>
              <button
                type="button"
                onClick={() => {
                  setConsensusFilter('exclusive')
                  setPage(0)
                }}
                className={`px-2 py-0.5 rounded text-2xs font-medium transition-all ${
                  consensusFilter === 'exclusive'
                    ? 'bg-surface text-slate-900 dark:text-white shadow-2xs font-bold'
                    : 'text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                Single Collection ({singleCollectionCount})
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Sort Selector */}
            <div className="flex items-center gap-1.5">
              <span className="text-2xs text-slate-500 dark:text-white/60">Sort:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="rounded-md border border-slate-300 dark:border-white/20 bg-surface dark:bg-[#1e293b] px-2 py-1 text-2xs font-semibold text-slate-800 dark:text-white shadow-2xs focus:outline-hidden focus:ring-1 focus:ring-indigo-500 cursor-pointer"
                aria-label="Sort patents by"
              >
                <option value="collections" className="bg-white text-slate-900 dark:bg-[#1e293b] dark:text-white">
                  Most Collections Matched
                </option>
                <option value="score" className="bg-white text-slate-900 dark:bg-[#1e293b] dark:text-white">
                  Highest Score
                </option>
                <option value="features" className="bg-white text-slate-900 dark:bg-[#1e293b] dark:text-white">
                  Most Features Matched
                </option>
                <option value="id" className="bg-white text-slate-900 dark:bg-[#1e293b] dark:text-white">
                  Patent ID
                </option>
              </select>
            </div>

            {/* Quick Search Input */}
            <div className="w-56">
              <TextField
                placeholder="Search patent ID..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value)
                  setPage(0)
                }}
                className="text-xs"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main Matrix Table */}
      <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs overflow-hidden">
        {/* Table Top Status Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-white/10 bg-slate-50/70 dark:bg-white/5 px-4 py-2.5 text-xs">
          <div className="text-slate-600 dark:text-white/70">
            Showing{' '}
            <strong className="font-mono text-slate-900 dark:text-white">
              {filteredRows.length === 0 ? 0 : safePage * pageSize + 1}–
              {Math.min(filteredRows.length, (safePage + 1) * pageSize)}
            </strong>{' '}
            of <strong className="font-mono text-slate-900 dark:text-white">{filteredRows.length}</strong> patents
            {filteredRows.length !== matrixData.rows.length && ` (${matrixData.rows.length} total)`}
          </div>

          {/* Quick Page Prev/Next on top */}
          {totalPages > 1 && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => handlePageChange(Math.max(0, safePage - 1))}
                disabled={safePage === 0}
                className="px-2 py-0.5 rounded text-2xs font-semibold bg-surface dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-white/20 transition-colors"
              >
                ‹ Prev
              </button>
              <span className="text-2xs font-mono font-medium text-slate-600 dark:text-white/70 px-1">
                {safePage + 1}/{totalPages}
              </span>
              <button
                type="button"
                onClick={() => handlePageChange(Math.min(totalPages - 1, safePage + 1))}
                disabled={safePage >= totalPages - 1}
                className="px-2 py-0.5 rounded text-2xs font-semibold bg-surface dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-white/20 transition-colors"
              >
                Next ›
              </button>
            </div>
          )}
        </div>

        {/* Scrollable Matrix Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-white/10 bg-slate-100/70 dark:bg-white/5 text-2xs font-bold uppercase tracking-wider text-slate-600 dark:text-white/70">
                {/* Left side: Patent info columns */}
                <th scope="col" className="px-3.5 py-3 w-12 text-center">
                  #
                </th>
                <th scope="col" className="px-3.5 py-3 min-w-[150px]">
                  Patent ID
                </th>
                <th scope="col" className="px-3.5 py-3 min-w-[110px] text-center">
                  Matched Features
                </th>
                <th scope="col" className="px-3.5 py-3 min-w-[120px] text-center">
                  Consensus
                </th>

                {/* Right side: All collection names in the header */}
                {series.map((s) => {
                  const count = matrixData.countPerCollection[s.name] ?? 0
                  return (
                    <th
                      key={s.name}
                      scope="col"
                      className="px-4 py-3 min-w-[130px] text-center border-l border-slate-200/80 dark:border-white/10 bg-slate-50/50 dark:bg-white/[0.02]"
                    >
                      <div className="flex flex-col items-center justify-center gap-0.5">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="size-2.5 rounded-xs shrink-0 shadow-2xs"
                            style={{ backgroundColor: s.color }}
                          />
                          <span className="font-bold font-mono text-slate-900 dark:text-white text-xs">
                            {s.name}
                          </span>
                        </div>
                        <span className="text-3xs font-mono font-normal text-slate-500 dark:text-white/60">
                          {count} result{count === 1 ? '' : 's'}
                        </span>
                      </div>
                    </th>
                  )
                })}
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-200 dark:divide-white/10 bg-surface">
              {pagedRows.length === 0 ? (
                <tr>
                  <td
                    colSpan={4 + series.length}
                    className="px-6 py-12 text-center text-xs text-slate-500 dark:text-white/60 italic"
                  >
                    No patents match the selected feature and filters.
                  </td>
                </tr>
              ) : (
                pagedRows.map((row, idx) => {
                  const url = patentUrl(config?.patent_view_url_template, row.patent_id)
                  const absoluteRank = safePage * pageSize + idx + 1
                  const isExpanded = expandedPatentId === row.patent_id
                  const isFullConsensus = row.matchedCollections.length === collectionNames.length

                  return (
                    <>
                      <tr
                        key={row.patent_id}
                        className={`transition-colors hover:bg-slate-50/80 dark:hover:bg-white/5 ${
                          isExpanded ? 'bg-indigo-50/30 dark:bg-indigo-950/15' : ''
                        }`}
                      >
                        {/* Rank */}
                        <td className="px-3.5 py-3 align-middle text-center text-slate-500 dark:text-white/50 font-mono text-xs">
                          #{absoluteRank}
                        </td>

                        {/* Patent ID with Expand toggle */}
                        <td className="px-3.5 py-3 align-middle">
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() =>
                                setExpandedPatentId(isExpanded ? null : row.patent_id)
                              }
                              className="size-5 rounded flex items-center justify-center text-slate-400 hover:text-slate-600 dark:hover:text-white transition-colors"
                              title={isExpanded ? 'Collapse chunks' : 'Expand chunks'}
                            >
                              <ChevronIcon
                                className={`size-3.5 transition-transform ${
                                  isExpanded ? 'rotate-90 text-indigo-600 dark:text-indigo-400' : ''
                                }`}
                              />
                            </button>

                            <div className="flex flex-col">
                              {url ? (
                                <a
                                  href={url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="font-mono font-bold text-indigo-600 hover:underline dark:text-indigo-300 inline-flex items-center gap-1"
                                >
                                  <span>{row.patent_id}</span>
                                  <ExternalIcon className="size-3 text-slate-400" />
                                </a>
                              ) : (
                                <span className="font-mono font-bold text-indigo-600 dark:text-indigo-300">
                                  {row.patent_id}
                                </span>
                              )}
                              <span className="text-3xs text-slate-400 dark:text-white/50">
                                {row.totalChunks} chunk{row.totalChunks === 1 ? '' : 's'}
                              </span>
                            </div>
                          </div>
                        </td>

                        {/* Matched Features */}
                        <td className="px-3.5 py-3 align-middle text-center">
                          <div className="flex flex-wrap items-center justify-center gap-1">
                            {row.matchedFeatureIds
                              .slice()
                              .sort((a, b) => a - b)
                              .map((fid) => (
                                <span key={fid} title={`Feature ${fid}`}>
                                  <Badge
                                    tone="brand"
                                    className="font-mono text-3xs font-bold"
                                  >
                                    F{fid}
                                  </Badge>
                                </span>
                              ))}
                          </div>
                        </td>

                        {/* Consensus Badge */}
                        <td className="px-3.5 py-3 align-middle text-center">
                          {isFullConsensus ? (
                            <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 text-2xs font-bold text-emerald-700 dark:text-emerald-300">
                              All Collections ({collectionNames.length})
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded-md bg-slate-100 dark:bg-white/10 px-2 py-0.5 text-2xs font-semibold text-slate-700 dark:text-white/80">
                              {row.matchedCollections.length} of {collectionNames.length}
                            </span>
                          )}
                        </td>

                        {/* Right side: Collection columns with Tik Mark (✓) */}
                        {series.map((s) => {
                          const colDetail = row.byCollection[s.name]
                          const isMatched = colDetail?.matched ?? false

                          return (
                            <td
                              key={s.name}
                              className="px-4 py-3 align-middle text-center border-l border-slate-200/80 dark:border-white/10"
                            >
                              {isMatched ? (
                                <div className="inline-flex flex-col items-center justify-center">
                                  {/* Prominent Green Tik Mark */}
                                  <span
                                    className="inline-flex size-6 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-400 font-bold shadow-2xs"
                                    title={`Result in ${s.name} with score ${colDetail.bestScore.toFixed(3)}`}
                                  >
                                    <CheckIcon className="size-3.5 stroke-[2.5]" />
                                  </span>

                                  {/* Score below the tik mark */}
                                  {colDetail.bestScore > 0 && (
                                    <span className="mt-0.5 font-mono text-[10.5px] font-bold text-slate-700 dark:text-white/90">
                                      {colDetail.bestScore.toFixed(3)}
                                    </span>
                                  )}

                                  {/* Feature tags if multiple in 'all' view */}
                                  {selectedFeatureId === 'all' &&
                                    colDetail.matchedFeatureIds.length > 0 && (
                                      <span className="text-3xs font-mono text-slate-400 dark:text-white/50">
                                        F{colDetail.matchedFeatureIds.join(', ')}
                                      </span>
                                    )}
                                </div>
                              ) : (
                                <div className="inline-flex items-center justify-center text-slate-300 dark:text-white/20 font-bold text-sm">
                                  —
                                </div>
                              )}
                            </td>
                          )
                        })}
                      </tr>

                      {/* Expanded Chunks Row */}
                      {isExpanded && (
                        <tr className="bg-slate-50/70 dark:bg-white/[0.02]">
                          <td
                            colSpan={4 + series.length}
                            className="px-6 py-4 border-y border-indigo-100 dark:border-indigo-900/30"
                          >
                            <div className="space-y-3">
                              <div className="flex items-center justify-between">
                                <h5 className="font-bold text-xs text-slate-800 dark:text-white flex items-center gap-2">
                                  <span>Chunks & Evidence for</span>
                                  <span className="font-mono text-indigo-600 dark:text-indigo-300">
                                    {row.patent_id}
                                  </span>
                                </h5>

                                <button
                                  type="button"
                                  onClick={() => setExpandedPatentId(null)}
                                  className="text-2xs text-slate-500 hover:text-slate-800 dark:hover:text-white underline cursor-pointer"
                                >
                                  Close
                                </button>
                              </div>

                              {/* Chunks Grouped by Collection */}
                              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                                {series.map((s) => {
                                  const colDetail = row.byCollection[s.name]
                                  const chunks = colDetail?.chunks || []

                                  return (
                                    <div
                                      key={s.name}
                                      className="rounded-lg border border-slate-200 dark:border-white/10 bg-surface p-3 space-y-2"
                                    >
                                      <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-white/5">
                                        <div className="flex items-center gap-1.5">
                                          <span
                                            className="size-2.5 rounded-xs shrink-0"
                                            style={{ backgroundColor: s.color }}
                                          />
                                          <span className="font-bold font-mono text-xs text-slate-800 dark:text-white">
                                            {s.name}
                                          </span>
                                        </div>
                                        {colDetail.matched ? (
                                          <Badge tone="success" className="text-3xs font-bold">
                                            ✓ Result ({colDetail.bestScore.toFixed(3)})
                                          </Badge>
                                        ) : (
                                          <span className="text-3xs text-slate-400 italic">No result</span>
                                        )}
                                      </div>

                                      {chunks.length === 0 ? (
                                        <p className="text-3xs text-slate-400 dark:text-white/40 italic py-2">
                                          No chunks returned in this collection.
                                        </p>
                                      ) : (
                                        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                                          {chunks.map((ch, chIdx) => (
                                            <div
                                              key={ch.chunk_id ?? chIdx}
                                              className="rounded border border-slate-100 dark:border-white/5 bg-slate-50/50 dark:bg-white/[0.03] p-2 space-y-1"
                                            >
                                              <div className="flex items-center justify-between text-3xs font-mono">
                                                <span className="text-indigo-600 dark:text-indigo-400 font-bold">
                                                  Chunk #{ch.chunk_id}
                                                </span>
                                                {ch.reranker_score != null && (
                                                  <Badge
                                                    tone={scoreTone(ch.reranker_score)}
                                                    className="font-mono text-3xs"
                                                  >
                                                    {ch.reranker_score.toFixed(3)}
                                                  </Badge>
                                                )}
                                              </div>
                                              {ch.section && (
                                                <div className="text-3xs text-slate-500 dark:text-white/60">
                                                  Section: <em>{ch.section}</em>
                                                </div>
                                              )}
                                              <p className="text-2xs text-slate-700 dark:text-white/80 line-clamp-3">
                                                {ch.text}
                                              </p>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Bottom Pagination Bar */}
        {filteredRows.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 dark:border-white/10 bg-slate-50/70 dark:bg-white/5 px-4 py-3 text-xs">
            {/* Left: Summary & Per-Page Selector */}
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-slate-600 dark:text-white/70">
                Showing{' '}
                <span className="font-semibold font-mono text-slate-900 dark:text-white">
                  {safePage * pageSize + 1}–{Math.min(filteredRows.length, (safePage + 1) * pageSize)}
                </span>{' '}
                of{' '}
                <span className="font-semibold font-mono text-slate-900 dark:text-white">
                  {filteredRows.length}
                </span>{' '}
                patents
              </span>

              <div className="flex items-center gap-1.5 pl-3 border-l border-slate-200 dark:border-white/10">
                <span className="text-2xs text-slate-500 dark:text-white/60">Per page:</span>
                <select
                  value={pageSize}
                  onChange={(e) => {
                    setPageSize(Number(e.target.value))
                    setPage(0)
                  }}
                  className="rounded-md border border-slate-300 dark:border-white/20 bg-surface dark:bg-[#1e293b] px-2.5 py-1 text-2xs font-semibold text-slate-800 dark:text-white shadow-2xs focus:outline-hidden focus:ring-1 focus:ring-indigo-500 cursor-pointer"
                  aria-label="Patents per page"
                >
                  {[15, 25, 50, 100, 200].map((size) => (
                    <option
                      key={size}
                      value={size}
                      className="bg-white text-slate-900 dark:bg-[#1e293b] dark:text-white"
                    >
                      {size}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Right: Page navigation buttons */}
            {totalPages > 1 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => handlePageChange(0)}
                  disabled={safePage === 0}
                  className="px-2 py-1 rounded text-2xs font-semibold bg-surface dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-white/20 transition-colors"
                  title="First page"
                >
                  « First
                </button>
                <button
                  type="button"
                  onClick={() => handlePageChange(Math.max(0, safePage - 1))}
                  disabled={safePage === 0}
                  className="px-2.5 py-1 rounded text-2xs font-semibold bg-surface dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-white/20 transition-colors"
                  title="Previous page"
                >
                  ‹ Prev
                </button>

                <div className="flex items-center gap-1 px-1 text-2xs font-mono font-medium text-slate-700 dark:text-white">
                  Page <strong className="font-bold">{safePage + 1}</strong> of{' '}
                  <strong className="font-bold">{totalPages}</strong>
                </div>

                <button
                  type="button"
                  onClick={() => handlePageChange(Math.min(totalPages - 1, safePage + 1))}
                  disabled={safePage >= totalPages - 1}
                  className="px-2.5 py-1 rounded text-2xs font-semibold bg-surface dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-white/20 transition-colors"
                  title="Next page"
                >
                  Next ›
                </button>
                <button
                  type="button"
                  onClick={() => handlePageChange(totalPages - 1)}
                  disabled={safePage >= totalPages - 1}
                  className="px-2 py-1 rounded text-2xs font-semibold bg-surface dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-white/20 transition-colors"
                  title="Last page"
                >
                  Last »
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

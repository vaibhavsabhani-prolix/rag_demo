import { useMemo, useState } from 'react'
import { Badge, SegmentedControl, TextField } from '@/components/ui'
import { ChevronIcon, ExternalIcon } from '@/components/ui/icons'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type { KeyFeatureItem, KeyFeatureSearchResponse, FeatureChunkResult } from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'

interface KeyFeatureMappingProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse; elapsedMs?: number; error?: string }>
  series: Series[]
}

interface FeatureMatchInfo {
  feature_id: number
  feature: string
  score: number
  rank?: number
  topChunk?: FeatureChunkResult
}

interface PatentFeatureMapping {
  patent_id: string
  title?: string
  matchedFeatures: FeatureMatchInfo[]
  featureCount: number
  bestScore: number
  avgScore: number
  otherCollections: string[]
}

// Distinct, vibrant visual palette for each feature number
const FEATURE_PALETTES = [
  {
    badge: 'bg-indigo-600 text-white',
    pill: 'bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-500/20 dark:text-indigo-200 dark:border-indigo-500/40',
    border: 'border-indigo-200 dark:border-indigo-800/60',
  },
  {
    badge: 'bg-emerald-600 text-white',
    pill: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/20 dark:text-emerald-200 dark:border-emerald-500/40',
    border: 'border-emerald-200 dark:border-emerald-800/60',
  },
  {
    badge: 'bg-amber-600 text-white',
    pill: 'bg-amber-50 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-200 dark:border-amber-500/40',
    border: 'border-amber-200 dark:border-amber-800/60',
  },
  {
    badge: 'bg-purple-600 text-white',
    pill: 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-500/20 dark:text-purple-200 dark:border-purple-500/40',
    border: 'border-purple-200 dark:border-purple-800/60',
  },
  {
    badge: 'bg-sky-600 text-white',
    pill: 'bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-500/20 dark:text-sky-200 dark:border-sky-500/40',
    border: 'border-sky-200 dark:border-sky-800/60',
  },
  {
    badge: 'bg-rose-600 text-white',
    pill: 'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/20 dark:text-rose-200 dark:border-rose-500/40',
    border: 'border-rose-200 dark:border-rose-800/60',
  },
  {
    badge: 'bg-teal-600 text-white',
    pill: 'bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-500/20 dark:text-teal-200 dark:border-teal-500/40',
    border: 'border-teal-200 dark:border-teal-800/60',
  },
  {
    badge: 'bg-orange-600 text-white',
    pill: 'bg-orange-50 text-orange-800 border-orange-200 dark:bg-orange-500/20 dark:text-orange-200 dark:border-orange-500/40',
    border: 'border-orange-200 dark:border-orange-800/60',
  },
]

function getFeaturePalette(featureId: number) {
  return FEATURE_PALETTES[(featureId - 1) % FEATURE_PALETTES.length]
}

function scoreTone(score: number): 'success' | 'warning' | 'danger' | 'neutral' {
  if (score >= 0.8) return 'success'
  if (score >= 0.6) return 'warning'
  if (score > 0) return 'danger'
  return 'neutral'
}

export function KeyFeatureMapping({ features, runs, series }: KeyFeatureMappingProps) {
  const { data: config } = usePipelineConfig()

  const [selectedCollection, setSelectedCollection] = useState<'all' | string>('all')
  const [minFeatures, setMinFeatures] = useState<number>(1)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeFeatureFilter, setActiveFeatureFilter] = useState<number | null>(null)
  const [expandedPatentKey, setExpandedPatentKey] = useState<string | null>(null)
  const [showFeatureLegend, setShowFeatureLegend] = useState(true)

  const featureTextById = useMemo(() => {
    const map = new Map<number, string>()
    for (const f of features) {
      map.set(f.feature_id, f.feature)
    }
    return map
  }, [features])

  // Extract patent-to-feature mapping for each collection
  const mappingByCollection = useMemo(() => {
    const allCollectionsMap: Record<string, PatentFeatureMapping[]> = {}
    const patentsPresence: Record<string, Set<string>> = {}

    // First pass: index presence across all collections
    for (const s of series) {
      patentsPresence[s.name] = new Set()
      const resp = runs[s.name]?.response
      if (!resp) continue

      for (const featRes of resp.results || []) {
        for (const p of featRes.patents || []) {
          patentsPresence[s.name].add(p.patent_id)
        }
        for (const ch of featRes.chunks || []) {
          patentsPresence[s.name].add(ch.patent_id)
        }
      }
    }

    // Second pass: build detailed patent mapping for each collection
    for (const s of series) {
      const collName = s.name
      const resp = runs[collName]?.response
      if (!resp) {
        allCollectionsMap[collName] = []
        continue
      }

      const patentMap = new Map<
        string,
        {
          patent_id: string
          title?: string
          featuresMap: Map<number, FeatureMatchInfo>
        }
      >()

      for (const featRes of resp.results || []) {
        const featureId = featRes.feature_id
        const featureText = featRes.feature || featureTextById.get(featureId) || `Feature ${featureId}`

        if (featRes.patents && featRes.patents.length > 0) {
          featRes.patents.forEach((p, rankIdx) => {
            let entry = patentMap.get(p.patent_id)
            if (!entry) {
              entry = {
                patent_id: p.patent_id,
                title: p.title || undefined,
                featuresMap: new Map(),
              }
              patentMap.set(p.patent_id, entry)
            }
            if (!entry.title && p.title) {
              entry.title = p.title
            }

            const sc = p.best_reranker_score ?? p.score
            const existing = entry.featuresMap.get(featureId)
            if (!existing || sc > existing.score) {
              entry.featuresMap.set(featureId, {
                feature_id: featureId,
                feature: featureText,
                score: sc,
                rank: rankIdx + 1,
                topChunk: p.chunks?.[0],
              })
            }
          })
        }

        for (const ch of featRes.chunks || []) {
          let entry = patentMap.get(ch.patent_id)
          if (!entry) {
            entry = {
              patent_id: ch.patent_id,
              title: ch.title || undefined,
              featuresMap: new Map(),
            }
            patentMap.set(ch.patent_id, entry)
          }
          if (!entry.title && ch.title) {
            entry.title = ch.title
          }

          const sc = ch.reranker_score ?? ch.score
          const existing = entry.featuresMap.get(featureId)
          if (!existing) {
            entry.featuresMap.set(featureId, {
              feature_id: featureId,
              feature: featureText,
              score: sc,
              topChunk: ch,
            })
          } else if (sc > existing.score) {
            existing.score = sc
            existing.topChunk = ch
          }
        }
      }

      const list: PatentFeatureMapping[] = Array.from(patentMap.values()).map((p) => {
        const matchedFeatures = Array.from(p.featuresMap.values()).sort(
          (a, b) => a.feature_id - b.feature_id
        )
        const scores = matchedFeatures.map((f) => f.score)
        const bestScore = scores.length > 0 ? Math.max(...scores) : 0
        const avgScore =
          scores.length > 0 ? scores.reduce((sum, s) => sum + s, 0) / scores.length : 0

        const otherCollections = series
          .map((col) => col.name)
          .filter((name) => name !== collName && patentsPresence[name]?.has(p.patent_id))

        return {
          patent_id: p.patent_id,
          title: p.title,
          matchedFeatures,
          featureCount: matchedFeatures.length,
          bestScore,
          avgScore,
          otherCollections,
        }
      })

      // Primary sort: Most features matched descending
      // Secondary sort: Best score descending
      list.sort((a, b) => b.featureCount - a.featureCount || b.bestScore - a.bestScore)

      allCollectionsMap[collName] = list
    }

    return allCollectionsMap
  }, [runs, series, featureTextById])

  // Filter helper
  const filterList = (list: PatentFeatureMapping[]) => {
    return list.filter((p) => {
      if (p.featureCount < minFeatures) return false
      if (activeFeatureFilter !== null && !p.matchedFeatures.some((f) => f.feature_id === activeFeatureFilter)) {
        return false
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim()
        const matchId = p.patent_id.toLowerCase().includes(q)
        const matchTitle = p.title?.toLowerCase().includes(q)
        const matchFeature = p.matchedFeatures.some((f) =>
          `f${f.feature_id}`.includes(q) || f.feature.toLowerCase().includes(q)
        )
        if (!matchId && !matchTitle && !matchFeature) return false
      }
      return true
    })
  }

  const toggleExpand = (key: string) => {
    setExpandedPatentKey((prev) => (prev === key ? null : key))
  }

  const collectionOptions = [
    { value: 'all', label: 'All Collections' },
    ...series.map((s) => ({ value: s.name, label: s.name })),
  ]

  const minFeatureOptions = [
    { value: '1', label: 'All' },
    { value: '2', label: '2+ Features' },
    { value: '3', label: '3+ Features' },
  ]

  return (
    <div className="space-y-5">
      {/* Feature Guide Banner */}
      <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-50/80 dark:bg-white/5 px-4 py-3 border-b border-slate-200 dark:border-white/10">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider">
              Key Technical Features ({features.length})
            </h3>
            <p className="text-xs text-slate-500 dark:text-white/70">
              Patents in each collection below are ranked by how many of these features they matched. Click any feature to filter.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {activeFeatureFilter !== null && (
              <button
                type="button"
                onClick={() => setActiveFeatureFilter(null)}
                className="px-2 py-1 rounded text-2xs font-semibold bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200 hover:bg-rose-200 transition-colors"
              >
                Clear F{activeFeatureFilter} Filter ✕
              </button>
            )}
            <button
              type="button"
              onClick={() => setShowFeatureLegend((v) => !v)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md border border-slate-200 dark:border-white/10 text-xs font-medium text-slate-600 dark:text-white hover:bg-slate-100 dark:hover:bg-white/10 transition-colors"
            >
              <span>{showFeatureLegend ? 'Hide' : 'Show'} Guide</span>
              <ChevronIcon
                className={`size-3 transition-transform ${showFeatureLegend ? 'rotate-90' : ''}`}
              />
            </button>
          </div>
        </div>

        {showFeatureLegend && (
          <div className="p-4 bg-surface">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
              {features.map((f) => {
                const pal = getFeaturePalette(f.feature_id)
                const isSelected = activeFeatureFilter === f.feature_id
                return (
                  <button
                    key={f.feature_id}
                    type="button"
                    onClick={() =>
                      setActiveFeatureFilter((prev) => (prev === f.feature_id ? null : f.feature_id))
                    }
                    className={`flex items-start gap-2.5 p-2 rounded-lg border text-left transition-all ${
                      isSelected
                        ? 'border-indigo-500 bg-indigo-50/80 dark:bg-indigo-950/40 shadow-xs ring-1 ring-indigo-500'
                        : 'border-slate-200/80 dark:border-white/10 hover:border-slate-300 dark:hover:border-white/20 bg-surface'
                    }`}
                  >
                    <span
                      className={`shrink-0 px-2 py-0.5 rounded font-mono font-bold text-xs shadow-2xs ${pal.badge}`}
                    >
                      F{f.feature_id}
                    </span>
                    <span className="text-2xs text-slate-700 dark:text-white leading-relaxed line-clamp-2">
                      {f.feature}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Filter and View Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 dark:border-white/10 bg-surface p-3 shadow-xs">
        <div className="flex flex-wrap items-center gap-3">
          {/* Collection View Switcher */}
          <div className="flex items-center gap-1.5">
            <span className="text-2xs font-bold uppercase tracking-wider text-slate-400 dark:text-white/60">
              View:
            </span>
            <SegmentedControl
              size="sm"
              value={selectedCollection}
              onChange={(val) => setSelectedCollection(val as string)}
              options={collectionOptions}
            />
          </div>

          {/* Min Features Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-2xs font-bold uppercase tracking-wider text-slate-400 dark:text-white/60">
              Matches:
            </span>
            <SegmentedControl
              size="sm"
              value={String(minFeatures)}
              onChange={(val) => setMinFeatures(Number(val))}
              options={minFeatureOptions}
            />
          </div>
        </div>

        {/* Quick Search Box */}
        <div className="w-full sm:w-64">
          <TextField
            placeholder="Search patent or feature..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="text-xs"
          />
        </div>
      </div>

      {/* Main Content: Side-by-Side or Single Collection */}
      {selectedCollection === 'all' ? (
        /* Side-by-Side Multi-Column View */
        <div
          className={`grid grid-cols-1 gap-5 ${
            series.length === 2
              ? 'md:grid-cols-2'
              : series.length >= 3
              ? 'lg:grid-cols-3'
              : ''
          }`}
        >
          {series.map((s) => {
            const rawList = mappingByCollection[s.name] || []
            const filtered = filterList(rawList)
            const multiCount = rawList.filter((p) => p.featureCount >= 2).length

            return (
              <div
                key={s.name}
                className="flex flex-col rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs overflow-hidden"
              >
                {/* Column Header */}
                <div className="flex items-center justify-between gap-2 border-b border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="size-3 rounded-xs shrink-0" style={{ backgroundColor: s.color }} />
                    <span className="font-bold text-base text-slate-900 dark:text-white">
                      {s.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Badge tone="neutral" className="font-mono text-2xs font-semibold dark:text-white">
                      {filtered.length} {filtered.length === 1 ? 'patent' : 'patents'}
                    </Badge>
                    {multiCount > 0 && (
                      <Badge tone="brand" className="font-mono text-2xs font-bold dark:text-white">
                        {multiCount} with 2+ feat
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Column Body: Ranked Patents List */}
                <div className="p-3 space-y-2.5 max-h-[44rem] overflow-y-auto flex-1">
                  {filtered.length === 0 ? (
                    <p className="p-8 text-center text-xs text-slate-400 dark:text-white/40 italic">
                      No patents match the current filter in “{s.name}”.
                    </p>
                  ) : (
                    filtered.map((patent, rankIdx) => {
                      const patentKey = `${s.name}-${patent.patent_id}`
                      const isExpanded = expandedPatentKey === patentKey
                      const url = patentUrl(config?.patent_view_url_template, patent.patent_id)
                      const isHighCoverage = patent.featureCount >= 3
                      const isMulti = patent.featureCount >= 2

                      return (
                        <div
                          key={patent.patent_id}
                          className={`rounded-lg border bg-surface transition-all ${
                            isHighCoverage
                              ? 'border-emerald-300 dark:border-emerald-600/50 shadow-2xs'
                              : isMulti
                              ? 'border-indigo-200 dark:border-indigo-800/50 shadow-2xs'
                              : 'border-slate-200 dark:border-white/10 hover:border-slate-300 dark:hover:border-white/20'
                          }`}
                        >
                          <div className="p-3 space-y-2">
                            {/* Patent Number & Stats Bar */}
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-2 min-w-0">
                                <span
                                  className={`flex size-5 items-center justify-center rounded font-mono text-2xs font-bold shrink-0 ${
                                    rankIdx === 0
                                      ? 'bg-amber-100 text-amber-900 dark:bg-amber-500/20 dark:text-amber-200 border border-amber-300 dark:border-amber-500/40'
                                      : 'bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-white border border-slate-200 dark:border-white/10'
                                  }`}
                                >
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
                                  <span className="font-mono text-xs font-bold text-slate-900 dark:text-white truncate">
                                    {patent.patent_id}
                                  </span>
                                )}
                              </div>

                              <div className="flex items-center gap-1.5 shrink-0">
                                <span
                                  className={`inline-flex items-center px-1.5 py-0.5 rounded font-mono text-2xs font-bold ${
                                    isHighCoverage
                                      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-500/40'
                                      : isMulti
                                      ? 'bg-indigo-100 text-indigo-800 dark:bg-indigo-500/20 dark:text-indigo-200 border border-indigo-300 dark:border-indigo-500/40'
                                      : 'bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-white border border-slate-200 dark:border-white/15'
                                  }`}
                                >
                                  {patent.featureCount} {patent.featureCount === 1 ? 'feature' : 'features'}
                                </span>

                                <Badge
                                  tone={scoreTone(patent.bestScore)}
                                  className="font-mono text-2xs font-bold dark:text-white"
                                >
                                  {patent.bestScore.toFixed(3)}
                                </Badge>
                              </div>
                            </div>

                            {/* Patent Title */}
                            {patent.title && (
                              <p
                                className="text-2xs text-slate-500 dark:text-white/70 line-clamp-1 leading-snug"
                                title={patent.title}
                              >
                                {patent.title}
                              </p>
                            )}

                            {/* Features Matched Row (e.g. F1 F5 F6) */}
                            <div className="flex flex-wrap items-center gap-1.5 pt-1">
                              {patent.matchedFeatures.map((f) => {
                                const pal = getFeaturePalette(f.feature_id)
                                return (
                                  <span
                                    key={f.feature_id}
                                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md font-mono text-xs font-bold border shadow-2xs transition-transform hover:scale-105 cursor-help ${pal.pill}`}
                                    title={`Feature ${f.feature_id}: ${f.feature}\nRerank Score: ${f.score.toFixed(3)}`}
                                  >
                                    <span>F{f.feature_id}</span>
                                    <span className="text-3xs font-semibold opacity-70 font-sans">
                                      {f.score.toFixed(2)}
                                    </span>
                                  </span>
                                )
                              })}
                            </div>

                            {/* Footer: Also In & Details Toggle */}
                            <div className="flex items-center justify-between border-t border-slate-200 dark:border-white/10 pt-2 text-2xs text-slate-500 dark:text-white/70">
                              <div>
                                {patent.otherCollections.length > 0 ? (
                                  <span className="inline-flex items-center gap-1 text-slate-500 dark:text-white/70 text-3xs">
                                    <span>Also in:</span>
                                    {patent.otherCollections.map((colName) => (
                                      <span
                                        key={colName}
                                        className="font-mono font-bold text-3xs px-1 rounded bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white border border-slate-200 dark:border-white/15"
                                      >
                                        {colName}
                                      </span>
                                    ))}
                                  </span>
                                ) : (
                                  <span className="text-3xs text-slate-400 dark:text-white/40 italic">
                                    Exclusive to {s.name}
                                  </span>
                                )}
                              </div>

                              <button
                                type="button"
                                onClick={() => toggleExpand(patentKey)}
                                className="inline-flex items-center gap-1 font-semibold text-indigo-600 hover:text-indigo-800 dark:text-indigo-300 dark:hover:text-white transition-colors"
                              >
                                <span>{isExpanded ? 'Hide' : 'Details'}</span>
                                <ChevronIcon
                                  className={`size-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                                />
                              </button>
                            </div>
                          </div>

                          {/* Expanded Feature Match Snippets */}
                          {isExpanded && (
                            <div className="border-t border-slate-200 dark:border-white/10 bg-slate-50/70 dark:bg-white/5 p-3 space-y-2 text-2xs">
                              {patent.matchedFeatures.map((f) => {
                                const pal = getFeaturePalette(f.feature_id)
                                return (
                                  <div
                                    key={f.feature_id}
                                    className="rounded border border-slate-200 dark:border-white/10 bg-surface p-2 shadow-2xs space-y-1.5"
                                  >
                                    <div className="flex items-center justify-between border-b border-slate-200 dark:border-white/10 pb-1">
                                      <div className="flex items-center gap-1.5 font-bold text-slate-800 dark:text-white">
                                        <span
                                          className={`px-1.5 py-0.5 rounded font-mono text-3xs font-bold ${pal.badge}`}
                                        >
                                          F{f.feature_id}
                                        </span>
                                        <span className="truncate max-w-[200px]" title={f.feature}>
                                          {f.feature}
                                        </span>
                                      </div>
                                      <Badge
                                        tone={scoreTone(f.score)}
                                        className="font-mono text-2xs font-bold dark:text-white"
                                      >
                                        Rerank: {f.score.toFixed(3)}
                                      </Badge>
                                    </div>

                                    {f.topChunk && (
                                      <p className="text-slate-600 dark:text-white/80 line-clamp-3 leading-relaxed italic">
                                        “{f.topChunk.text}”
                                      </p>
                                    )}
                                  </div>
                                )
                              })}
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
      ) : (
        /* Single Collection Full-Width Table View */
        (() => {
          const rawList = mappingByCollection[selectedCollection] || []
          const filtered = filterList(rawList)
          const seriesItem = series.find((s) => s.name === selectedCollection)

          return (
            <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 px-4 py-3">
                <div className="flex items-center gap-2">
                  {seriesItem && (
                    <span
                      className="size-3 rounded-xs shrink-0"
                      style={{ backgroundColor: seriesItem.color }}
                    />
                  )}
                  <h4 className="font-bold text-sm text-slate-900 dark:text-white">
                    Collection “{selectedCollection}” — Feature Coverage Table
                  </h4>
                </div>
                <Badge tone="neutral" className="font-mono text-2xs font-semibold dark:text-white">
                  Showing {filtered.length} of {rawList.length} patents
                </Badge>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 dark:divide-white/10 text-xs">
                  <thead className="bg-slate-50 dark:bg-surface">
                    <tr>
                      <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs w-12">
                        Rank
                      </th>
                      <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs min-w-[200px]">
                        Patent ID & Title
                      </th>
                      <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                        Matched Features
                      </th>
                      <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                        Feature Count
                      </th>
                      <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                        Best Score
                      </th>
                      <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                        Avg Score
                      </th>
                      <th className="px-3.5 py-2.5 text-right font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs w-20">
                        Details
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-white/10 bg-surface">
                    {filtered.length === 0 ? (
                      <tr>
                        <td
                          colSpan={7}
                          className="p-8 text-center text-xs text-slate-400 dark:text-white/40 italic"
                        >
                          No patents matched the current filter criteria in {selectedCollection}.
                        </td>
                      </tr>
                    ) : (
                      filtered.map((patent, rankIdx) => {
                        const patentKey = `${selectedCollection}-${patent.patent_id}`
                        const isExpanded = expandedPatentKey === patentKey
                        const url = patentUrl(config?.patent_view_url_template, patent.patent_id)

                        return (
                          <>
                            <tr
                              key={patent.patent_id}
                              className="hover:bg-slate-50/70 dark:hover:bg-white/5 transition-colors"
                            >
                              <td className="px-3.5 py-3 align-middle text-center">
                                <span className="font-mono text-xs font-bold text-slate-700 dark:text-white">
                                  #{rankIdx + 1}
                                </span>
                              </td>

                              <td className="px-3.5 py-3 align-middle">
                                <div className="flex flex-col">
                                  {url ? (
                                    <a
                                      href={url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="font-mono text-xs font-bold text-indigo-600 hover:underline dark:text-white inline-flex items-center gap-1"
                                    >
                                      <span>{patent.patent_id}</span>
                                      <ExternalIcon className="size-2.5 opacity-60" />
                                    </a>
                                  ) : (
                                    <span className="font-mono text-xs font-bold text-slate-900 dark:text-white">
                                      {patent.patent_id}
                                    </span>
                                  )}
                                  {patent.title && (
                                    <span
                                      className="text-2xs text-slate-500 dark:text-white/70 truncate max-w-sm mt-0.5"
                                      title={patent.title}
                                    >
                                      {patent.title}
                                    </span>
                                  )}
                                </div>
                              </td>

                              <td className="px-3.5 py-3 align-middle">
                                <div className="flex flex-wrap items-center gap-1.5">
                                  {patent.matchedFeatures.map((f) => {
                                    const pal = getFeaturePalette(f.feature_id)
                                    return (
                                      <span
                                        key={f.feature_id}
                                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md font-mono text-xs font-bold border ${pal.pill}`}
                                        title={`Feature ${f.feature_id}: ${f.feature}\nRerank Score: ${f.score.toFixed(3)}`}
                                      >
                                        <span>F{f.feature_id}</span>
                                        <span className="text-3xs font-semibold opacity-70 font-sans">
                                          {f.score.toFixed(2)}
                                        </span>
                                      </span>
                                    )
                                  })}
                                </div>
                              </td>

                              <td className="px-3.5 py-3 align-middle text-center">
                                <Badge
                                  tone={patent.featureCount >= 3 ? 'success' : patent.featureCount >= 2 ? 'brand' : 'neutral'}
                                  className="font-mono text-2xs font-bold dark:text-white"
                                >
                                  {patent.featureCount} / {features.length}
                                </Badge>
                              </td>

                              <td className="px-3.5 py-3 align-middle text-center">
                                <Badge
                                  tone={scoreTone(patent.bestScore)}
                                  className="font-mono text-2xs font-bold dark:text-white"
                                >
                                  {patent.bestScore.toFixed(3)}
                                </Badge>
                              </td>

                              <td className="px-3.5 py-3 align-middle text-center">
                                <span className="font-mono text-xs text-slate-600 dark:text-white/80">
                                  {patent.avgScore.toFixed(3)}
                                </span>
                              </td>

                              <td className="px-3.5 py-3 align-middle text-right">
                                <button
                                  type="button"
                                  onClick={() => toggleExpand(patentKey)}
                                  className="inline-flex items-center gap-1 text-2xs font-medium text-indigo-600 hover:text-indigo-800 dark:text-indigo-300 dark:hover:text-white"
                                >
                                  <span>{isExpanded ? 'Hide' : 'Details'}</span>
                                  <ChevronIcon
                                    className={`size-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                                  />
                                </button>
                              </td>
                            </tr>

                            {isExpanded && (
                              <tr className="bg-slate-50/70 dark:bg-white/5">
                                <td colSpan={7} className="px-6 py-4">
                                  <div className="space-y-3">
                                    <div className="text-2xs font-bold uppercase tracking-wider text-slate-500 dark:text-white/70">
                                      Matched Technical Features & Snippets for {patent.patent_id}
                                    </div>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                      {patent.matchedFeatures.map((f) => {
                                        const pal = getFeaturePalette(f.feature_id)
                                        return (
                                          <div
                                            key={f.feature_id}
                                            className="rounded-lg border border-slate-200 dark:border-white/10 bg-surface p-3 space-y-1.5 shadow-2xs"
                                          >
                                            <div className="flex items-center justify-between border-b border-slate-200 dark:border-white/10 pb-1.5">
                                              <div className="flex items-center gap-1.5">
                                                <span
                                                  className={`px-1.5 py-0.5 rounded font-mono font-bold text-3xs ${pal.badge}`}
                                                >
                                                  F{f.feature_id}
                                                </span>
                                                <span className="font-semibold text-xs text-slate-800 dark:text-white">
                                                  Feature {f.feature_id}
                                                </span>
                                              </div>
                                              <Badge
                                                tone={scoreTone(f.score)}
                                                className="font-mono text-2xs font-bold dark:text-white"
                                              >
                                                Rerank: {f.score.toFixed(3)}
                                              </Badge>
                                            </div>
                                            <p className="text-2xs text-slate-600 dark:text-white/80 font-medium">
                                              {f.feature}
                                            </p>
                                            {f.topChunk && (
                                              <div className="mt-1 p-2 rounded bg-slate-50 dark:bg-white/5 border border-slate-200/60 dark:border-white/5 text-2xs text-slate-700 dark:text-white/90 leading-relaxed max-h-28 overflow-y-auto italic">
                                                “{f.topChunk.text}”
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
            </div>
          )
        })()
      )}
    </div>
  )
}

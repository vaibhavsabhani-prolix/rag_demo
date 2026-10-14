import { useMemo, useState } from 'react'
import { Badge, SegmentedControl } from '@/components/ui'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type { KeyFeatureItem, KeyFeatureSearchResponse } from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'

interface PatentConsensusTableProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse }>
  series: Series[]
}

interface ConsensusPatentRow {
  patent_id: string
  title?: string
  bestScore: number
  matchedCollections: string[]
  matchedFeatureIds: number[]
  totalChunks: number
}

export function PatentConsensusTable({ features: _features, runs, series }: PatentConsensusTableProps) {
  const { data: config } = usePipelineConfig()
  const [filterMode, setFilterMode] = useState<'all' | 'all_collections' | 'consensus' | 'exclusive'>('all')

  const collectionNames = series.map((s) => s.name)
  const seriesByCollection = Object.fromEntries(series.map((s) => [s.name, s]))

  const rows = useMemo<ConsensusPatentRow[]>(() => {
    const map = new Map<string, ConsensusPatentRow>()

    for (const collName of collectionNames) {
      const resp = runs[collName]?.response
      if (!resp) continue

      for (const featRes of resp.results || []) {
        // Collect chunks and patents
        for (const chunk of featRes.chunks || []) {
          let row = map.get(chunk.patent_id)
          if (!row) {
            row = {
              patent_id: chunk.patent_id,
              title: chunk.title ?? undefined,
              bestScore: chunk.score,
              matchedCollections: [],
              matchedFeatureIds: [],
              totalChunks: 0,
            }
            map.set(chunk.patent_id, row)
          }

          if (!row.title && chunk.title) row.title = chunk.title
          const sc = chunk.reranker_score ?? chunk.score
          if (sc > row.bestScore) row.bestScore = sc
          row.totalChunks += 1

          if (!row.matchedCollections.includes(collName)) {
            row.matchedCollections.push(collName)
          }
          if (!row.matchedFeatureIds.includes(featRes.feature_id)) {
            row.matchedFeatureIds.push(featRes.feature_id)
          }
        }
      }
    }

    return Array.from(map.values()).sort(
      (a, b) =>
        b.matchedCollections.length - a.matchedCollections.length ||
        b.matchedFeatureIds.length - a.matchedFeatureIds.length ||
        b.bestScore - a.bestScore,
    )
  }, [runs, collectionNames])

  const filteredRows = rows.filter((r) => {
    if (filterMode === 'all_collections') return r.matchedCollections.length === collectionNames.length
    if (filterMode === 'consensus') return r.matchedCollections.length >= 2
    if (filterMode === 'exclusive') return r.matchedCollections.length === 1
    return true
  })

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-xs space-y-0">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-slate-50/80 p-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wide">
            Cross-Collection Patent Consensus ({rows.length} unique patents)
          </h3>
          <p className="text-xs text-slate-500 dark:text-white/70">
            Patents discovered across collections, ranked by how many collections and features they satisfy.
          </p>
        </div>

        <SegmentedControl
          label="Filter"
          value={filterMode}
          onChange={(val) => setFilterMode(val as 'all' | 'all_collections' | 'consensus' | 'exclusive')}
          options={[
            { value: 'all', label: `All (${rows.length})` },
            {
              value: 'all_collections',
              label: `All Collections (${rows.filter((r) => r.matchedCollections.length === collectionNames.length).length})`,
            },
            {
              value: 'consensus',
              label: `2+ Collections (${rows.filter((r) => r.matchedCollections.length >= 2).length})`,
            },
            {
              value: 'exclusive',
              label: `Single Collection (${rows.filter((r) => r.matchedCollections.length === 1).length})`,
            },
          ]}
        />
      </div>

      <div className="max-h-[32rem] overflow-y-auto">
        {filteredRows.length === 0 ? (
          <p className="p-6 text-center text-xs text-slate-500 italic">No patents match this filter.</p>
        ) : (
          <table className="min-w-full divide-y divide-slate-200 text-xs">
            <thead className="sticky top-0 bg-slate-50">
              <tr>
                <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 uppercase">Patent ID</th>
                <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 uppercase min-w-[200px]">
                  Title
                </th>
                <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 uppercase">
                  Matched Collections
                </th>
                <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 uppercase">
                  Features Matched
                </th>
                <th className="px-3.5 py-2.5 text-right font-semibold text-slate-500 uppercase">
                  Top Score
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-surface">
              {filteredRows.map((row) => {
                const url = patentUrl(config?.patent_view_url_template, row.patent_id)
                const isFullConsensus = row.matchedCollections.length === collectionNames.length

                return (
                  <tr key={row.patent_id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="px-3.5 py-2.5 align-top">
                      {url ? (
                        <a
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-mono font-bold text-indigo-600 hover:underline dark:text-indigo-300"
                        >
                          {row.patent_id}
                        </a>
                      ) : (
                        <span className="font-mono font-bold text-indigo-600 dark:text-indigo-300">{row.patent_id}</span>
                      )}
                      <div className="text-2xs text-slate-400 dark:text-white/60">
                        {row.totalChunks} total chunk{row.totalChunks === 1 ? '' : 's'}
                      </div>
                    </td>

                    <td className="px-3.5 py-2.5 align-top">
                      <span className="text-slate-900 dark:text-white line-clamp-2" title={row.title || ''}>
                        {row.title || <span className="text-slate-400 dark:text-white/50 italic">No title</span>}
                      </span>
                    </td>

                    <td className="px-3.5 py-2.5 align-top">
                      <div className="flex flex-wrap items-center gap-1.5">
                        {row.matchedCollections.map((colName) => {
                          const colColor = seriesByCollection[colName]?.color || '#6366f1'
                          return (
                            <span
                              key={colName}
                              className="inline-flex items-center gap-1 rounded border border-slate-200 dark:border-white/15 px-1.5 py-0.5 text-2xs font-semibold bg-surface text-slate-800 dark:text-white"
                            >
                              <span className="size-2 rounded-xs" style={{ backgroundColor: colColor }} />
                              {colName}
                            </span>
                          )
                        })}
                        {isFullConsensus && (
                          <Badge tone="success" className="text-2xs font-bold">
                            All Collections
                          </Badge>
                        )}
                      </div>
                    </td>

                    <td className="px-3.5 py-2.5 align-top">
                      <div className="flex flex-wrap gap-1">
                        {row.matchedFeatureIds
                          .slice()
                          .sort((a, b) => a - b)
                          .map((fid) => (
                            <Badge key={fid} tone="brand" className="font-mono text-2xs">
                              F{fid}
                            </Badge>
                          ))}
                      </div>
                    </td>

                    <td className="px-3.5 py-2.5 align-top text-right font-mono font-bold text-slate-900 dark:text-white">
                      {row.bestScore.toFixed(3)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

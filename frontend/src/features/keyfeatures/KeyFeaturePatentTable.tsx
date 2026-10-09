import { useMemo, useState } from 'react'
import { Badge, Checkbox } from '@/components/ui'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type { FeatureSearchResult } from '@/schemas/keyfeature'

interface KeyFeaturePatentTableProps {
  results: FeatureSearchResult[]
}

interface PatentRow {
  patent_id: string
  title?: string
  bestScore: number
  chunkCount: number
  matchedFeatureIds: number[]
}

function buildPatentRows(results: FeatureSearchResult[]): PatentRow[] {
  const rowsByPatent = new Map<string, PatentRow>()

  for (const result of results) {
    for (const chunk of result.chunks || []) {
      let row = rowsByPatent.get(chunk.patent_id)
      if (!row) {
        row = {
          patent_id: chunk.patent_id,
          title: chunk.title ?? undefined,
          bestScore: chunk.score,
          chunkCount: 0,
          matchedFeatureIds: [],
        }
        rowsByPatent.set(chunk.patent_id, row)
      }
      if (!row.title && chunk.title) row.title = chunk.title
      row.bestScore = Math.max(row.bestScore, chunk.score)
      row.chunkCount += 1
      if (!row.matchedFeatureIds.includes(result.feature_id)) {
        row.matchedFeatureIds.push(result.feature_id)
      }
    }
  }

  return Array.from(rowsByPatent.values()).sort(
    (a, b) => b.matchedFeatureIds.length - a.matchedFeatureIds.length || b.bestScore - a.bestScore,
  )
}

export function KeyFeaturePatentTable({ results }: KeyFeaturePatentTableProps) {
  const [checkedFeatureIds, setCheckedFeatureIds] = useState<Set<number>>(new Set())
  const { data: config } = usePipelineConfig()

  const patentRows = useMemo(() => buildPatentRows(results), [results])

  const filteredRows =
    checkedFeatureIds.size === 0
      ? patentRows
      : patentRows.filter((row) => row.matchedFeatureIds.some((fid) => checkedFeatureIds.has(fid)))

  const toggleFeatureChecked = (featureId: number) => {
    setCheckedFeatureIds((prev) => {
      const next = new Set(prev)
      if (next.has(featureId)) next.delete(featureId)
      else next.add(featureId)
      return next
    })
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
      {/* Left: Key Features list */}
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-xs">
        <div className="grid grid-cols-[3rem_1fr_3rem] items-center gap-2 border-b border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
          <span>Sr No</span>
          <span>Key Features</span>
          <span className="text-right">Filter</span>
        </div>
        <div className="max-h-[32rem] divide-y divide-slate-100 overflow-y-auto">
          {results.map((result, idx) => {
            const isChecked = checkedFeatureIds.has(result.feature_id)
            return (
              <button
                key={result.feature_id}
                type="button"
                onClick={() => toggleFeatureChecked(result.feature_id)}
                className={`grid w-full grid-cols-[3rem_1fr_3rem] items-start gap-2 px-3 py-3 text-left text-sm transition-colors ${
                  isChecked ? 'bg-indigo-50' : idx % 2 === 0 ? 'bg-surface hover:bg-slate-50' : 'bg-slate-50/50 hover:bg-slate-100/60'
                }`}
              >
                <span className="pt-0.5 font-mono text-xs text-slate-400">{idx + 1}</span>
                <span className={isChecked ? 'font-medium text-indigo-900' : 'text-slate-700'}>
                  {result.feature_id}. {result.feature}
                </span>
                <span className="flex justify-end pt-0.5" onClick={(e) => e.stopPropagation()}>
                  <Checkbox
                    label=""
                    checked={isChecked}
                    onChange={() => toggleFeatureChecked(result.feature_id)}
                  />
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Right: Patent results table */}
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-xs">
        <div className="flex items-center justify-between gap-2 border-b border-slate-200 bg-slate-50 px-3 py-2">
          <span className="text-xs font-semibold tracking-wide text-slate-500 uppercase">
            Patent Results {checkedFeatureIds.size > 0 && `· ${checkedFeatureIds.size} feature${checkedFeatureIds.size === 1 ? '' : 's'} selected`}
          </span>
          {checkedFeatureIds.size > 0 && (
            <button
              type="button"
              onClick={() => setCheckedFeatureIds(new Set())}
              className="text-xs font-medium text-indigo-600 hover:underline"
            >
              Clear filter
            </button>
          )}
        </div>
        <div className="max-h-[32rem] overflow-y-auto">
          {filteredRows.length === 0 ? (
            <p className="p-4 text-sm italic text-slate-500">No patents matched.</p>
          ) : (
            <table className="min-w-full divide-y divide-slate-100 text-sm">
              <thead className="sticky top-0 bg-slate-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-semibold tracking-wide text-slate-500 uppercase">
                    Publication Number
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-semibold tracking-wide text-slate-500 uppercase">
                    Best Score
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-semibold tracking-wide text-slate-500 uppercase">
                    Matched On (Feature IDs)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredRows.map((row) => {
                  const url = patentUrl(config?.patent_view_url_template, row.patent_id)
                  return (
                  <tr key={row.patent_id} className="hover:bg-slate-50">
                    <td className="px-3 py-2.5 align-top">
                      {url ? (
                        <a
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-mono text-sm font-semibold text-indigo-600 hover:underline"
                        >
                          {row.patent_id}
                        </a>
                      ) : (
                        <span className="font-mono text-sm font-semibold text-indigo-600">{row.patent_id}</span>
                      )}
                      <div className="mt-0.5 text-2xs text-slate-400">{row.chunkCount} matching chunk{row.chunkCount === 1 ? '' : 's'}</div>
                    </td>
                    <td className="px-3 py-2.5 align-top font-mono text-xs font-bold text-slate-800">
                      {row.bestScore.toFixed(3)}
                    </td>
                    <td className="px-3 py-2.5 align-top">
                      <div className="flex flex-wrap gap-1">
                        {row.matchedFeatureIds
                          .slice()
                          .sort((a, b) => a - b)
                          .map((fid) => (
                            <Badge
                              key={fid}
                              tone={checkedFeatureIds.has(fid) ? 'brand' : 'success'}
                              className="font-mono text-2xs"
                            >
                              {fid}
                            </Badge>
                          ))}
                      </div>
                    </td>
                  </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}

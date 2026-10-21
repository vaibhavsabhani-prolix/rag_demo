import { useState, useMemo } from 'react'
import { Card, CardBody, CardHeader, Stat, StatGrid, Tabs } from '@/components/ui'
import type { KeyFeatureItem, KeyFeatureSearchResponse } from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'
import { FeatureWiseMatrix } from './FeatureWiseMatrix'
import { PatentConsensusTable } from './PatentConsensusTable'
import { KeyFeatureMapping } from './KeyFeatureMapping'
import { StepWiseFilterView } from './StepWiseFilterView'
import { PatentResultsMatrix } from './PatentResultsMatrix'

interface FeatureCompareDashboardProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse; elapsedMs?: number; error?: string }>
  series: Series[]
  totalElapsedMs?: number | null
}

export function FeatureCompareDashboard({
  features,
  runs,
  series,
  totalElapsedMs,
}: FeatureCompareDashboardProps) {
  const [activeTab, setActiveTab] = useState<'matrix' | 'mapping' | 'steps' | 'results-matrix' | 'consensus' | 'timings'>('matrix')

  const collectionNames = series.map((s) => s.name)

  // Global aggregate statistics
  const stats = useMemo(() => {
    const allPatents = new Set<string>()
    const patentsPerCollection: Record<string, Set<string>> = {}

    for (const name of collectionNames) {
      patentsPerCollection[name] = new Set()
      const resp = runs[name]?.response
      if (!resp) continue

      for (const feat of resp.results || []) {
        if (feat.patents && feat.patents.length > 0) {
          for (const p of feat.patents) {
            allPatents.add(p.patent_id)
            patentsPerCollection[name].add(p.patent_id)
          }
        }
        for (const ch of feat.chunks || []) {
          allPatents.add(ch.patent_id)
          patentsPerCollection[name].add(ch.patent_id)
        }
      }
    }

    return {
      totalFeatures: features.length,
      patentsPerCollection,
      totalUniquePatents: allPatents.size,
    }
  }, [features, runs, collectionNames])

  const tabItems: { id: 'matrix' | 'mapping' | 'steps' | 'results-matrix' | 'consensus' | 'timings'; label: string }[] = [
    { id: 'matrix', label: 'Feature-Wise Comparison' },
    { id: 'mapping', label: 'Key Feature Mapping' },
    { id: 'steps', label: 'Step-Wise Filtering' },
    { id: 'results-matrix', label: 'Patent Results Matrix' },
    { id: 'consensus', label: 'Patent Consensus' },
    { id: 'timings', label: 'Timings & Execution' },
  ]

  return (
    <div className="space-y-6">
      {/* Top Aggregate Stats */}
      <StatGrid
        className={
          series.length + 1 === 3
            ? 'md:grid-cols-3'
            : series.length + 1 === 2
            ? 'md:grid-cols-2'
            : undefined
        }
      >
        <Stat label="Key Features" value={stats.totalFeatures} hint="Extracted by LLM" />
        {series.map((s) => {
          const count = stats.patentsPerCollection[s.name]?.size ?? 0
          return (
            <Stat
              key={s.name}
              label={
                <div className="flex items-center gap-1.5">
                  <span className="size-2 rounded-xs shrink-0" style={{ backgroundColor: s.color }} />
                  <span>{s.name} Unique Patents</span>
                </div>
              }
              value={count}
              hint="Overall across all features"
            />
          )
        })}
      </StatGrid>

      {/* Navigation Tabs */}
      <div className="border-b border-slate-200 dark:border-white/10">
        <Tabs
          items={tabItems}
          value={activeTab}
          onChange={(id) => setActiveTab(id)}
        />
      </div>

      {/* Tab Panels */}
      {activeTab === 'matrix' && (
        <FeatureWiseMatrix features={features} runs={runs} series={series} />
      )}

      {activeTab === 'mapping' && (
        <KeyFeatureMapping features={features} runs={runs} series={series} />
      )}

      {activeTab === 'steps' && (
        <StepWiseFilterView features={features} runs={runs} series={series} />
      )}

      {activeTab === 'results-matrix' && (
        <PatentResultsMatrix features={features} runs={runs} series={series} />
      )}


      {activeTab === 'consensus' && (
        <PatentConsensusTable features={features} runs={runs} series={series} />
      )}

      {activeTab === 'timings' && (() => {
        const firstWithTimings = Object.values(runs).find((r) => r?.response?.timings)?.response
        const sharedTimings = firstWithTimings?.timings
        const sharedExtractionMs = sharedTimings?.extraction_ms
        const sharedEmbeddingMs = sharedTimings?.embedding_ms

        return (
          <Card>
            <CardHeader
              title="Collection Execution Timings"
              subtitle="Feature extraction and embeddings run once globally, followed by collection-specific search, rerank, and enrichment."
            />
            <CardBody>
              {/* Shared One-Time Phase Banner */}
              <div className="mb-4 rounded-xl border border-indigo-100 bg-indigo-50/50 p-4 dark:border-indigo-900/40 dark:bg-indigo-950/20">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="size-2 rounded-full bg-emerald-500" />
                      <span className="text-xs font-bold uppercase tracking-wider text-indigo-950 dark:text-indigo-200">
                        Shared Phase (Runs Exactly Once)
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-white/80">
                      Features are extracted once from the invention text and converted to vector embeddings. Each collection below is searched and reranked using these shared feature embeddings.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2.5 text-xs font-mono">
                    <div className="rounded-lg bg-surface px-3 py-2 border border-indigo-200 dark:border-white/10 shadow-2xs">
                      <span className="text-slate-500 dark:text-white/60">LLM Feature Extraction: </span>
                      <span className="font-bold text-slate-900 dark:text-white">
                        {sharedExtractionMs != null
                          ? sharedExtractionMs === 0
                            ? 'Reused (0ms · LLM Skipped)'
                            : `${Math.round(sharedExtractionMs)}ms (Run Once)`
                          : '—'}
                      </span>
                    </div>
                    <div className="rounded-lg bg-surface px-3 py-2 border border-indigo-200 dark:border-white/10 shadow-2xs">
                      <span className="text-slate-500 dark:text-white/60">Feature Embeddings: </span>
                      <span className="font-bold text-slate-900 dark:text-white">
                        {sharedEmbeddingMs != null ? `${Math.round(sharedEmbeddingMs)}ms (Run Once)` : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 dark:divide-white/10 text-xs">
                  <thead className="bg-slate-50 dark:bg-surface">
                    <tr>
                      <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase">
                        Collection
                      </th>
                      <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase">
                        Qdrant Search
                      </th>
                      <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase">
                        BGE Reranking
                      </th>
                      <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase">
                        Enrichment
                      </th>
                      <th className="px-3.5 py-2.5 text-right font-semibold text-slate-700 dark:text-white uppercase">
                        Collection Total
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-white/10 bg-surface">
                    {series.map((s) => {
                      const resp = runs[s.name]?.response
                      const timings = resp?.timings || {}
                      const runElapsed = runs[s.name]?.elapsedMs

                      return (
                        <tr key={s.name} className="hover:bg-slate-50/70 dark:hover:bg-white/5 transition-colors">
                          <td className="px-3.5 py-3 font-semibold text-slate-900 dark:text-white">
                            <div className="flex items-center gap-2">
                              <span className="size-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                              <span>{s.name}</span>
                            </div>
                          </td>
                          <td className="px-3.5 py-3 font-mono text-slate-600 dark:text-white/80">
                            {timings.retrieval_ms ? `${Math.round(timings.retrieval_ms)}ms` : '—'}
                          </td>
                          <td className="px-3.5 py-3 font-mono text-slate-600 dark:text-white/80">
                            {timings.reranking_ms ? `${Math.round(timings.reranking_ms)}ms` : '—'}
                          </td>
                          <td className="px-3.5 py-3 font-mono text-slate-600 dark:text-white/80">
                            {timings.metadata_enrich_ms ? `${Math.round(timings.metadata_enrich_ms)}ms` : '—'}
                          </td>
                          <td className="px-3.5 py-3 font-mono font-bold text-slate-900 dark:text-white text-right">
                            {runElapsed
                              ? `${Math.round(runElapsed)}ms`
                              : timings.collection_total_ms
                              ? `${Math.round(timings.collection_total_ms)}ms`
                              : timings.total_ms
                              ? `${Math.round(timings.total_ms)}ms`
                              : '—'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {totalElapsedMs != null && (
                <div className="mt-4 pt-3 border-t border-slate-200 dark:border-white/10 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-white/70">
                  <span>Collections were queried and reranked using the single set of extracted features.</span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white">
                    Total Session Time: {(totalElapsedMs / 1000).toFixed(2)}s
                  </span>
                </div>
              )}
            </CardBody>
          </Card>
        )
      })()}
    </div>
  )
}

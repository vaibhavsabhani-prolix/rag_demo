import { useState } from 'react'
import { Badge, SegmentedControl } from '@/components/ui'
import type { KeyFeatureSearchResponse, FeatureSearchResult } from '@/schemas/keyfeature'
import { KeyFeatureCard } from './KeyFeatureCard'
import { KeyFeaturePatentTable } from './KeyFeaturePatentTable'

interface KeyFeatureResultsViewProps {
  response: KeyFeatureSearchResponse
}

export function KeyFeatureResultsView({ response }: KeyFeatureResultsViewProps) {
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | 'all'>('all')
  const [viewMode, setViewMode] = useState<'patent' | 'feature'>('patent')
  const results = response.results || []
  const totalChunks = results.reduce((sum, r) => sum + (r.chunks?.length || 0), 0)

  const displayedResults =
    selectedFeatureId === 'all'
      ? results
      : results.filter((r) => r.feature_id === selectedFeatureId)

  return (
    <div className="space-y-6">
      {/* Overview & Timings Banner */}
      <div className="rounded-xl border border-slate-200 bg-surface p-5 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-slate-900">Search Results by Feature</h2>
              <Badge tone="brand" className="text-xs font-semibold">
                {results.length} Features Extracted
              </Badge>
              <Badge tone="success" className="text-xs font-semibold">
                {totalChunks} Chunks Retrieved
              </Badge>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Collection: <span className="font-medium text-slate-700">{response.collection}</span>
              {response.score_threshold != null && (
                <> · Threshold: <span className="font-medium text-slate-700">≥ {response.score_threshold}</span></>
              )}
              {' '}· Each feature searched independently
            </p>
          </div>

          {/* Timings */}
          <div className="flex flex-wrap items-center gap-3 font-mono text-xs text-slate-600">
            {response.timings.extraction_ms && (
              <div className="rounded-md bg-slate-50 px-2.5 py-1.5 border border-slate-200">
                <span className="text-slate-400">LLM Extract:</span>{' '}
                <span className="font-bold text-slate-800">{Math.round(response.timings.extraction_ms)}ms</span>
              </div>
            )}
            {response.timings.embedding_ms && (
              <div className="rounded-md bg-slate-50 px-2.5 py-1.5 border border-slate-200">
                <span className="text-slate-400">Embeddings:</span>{' '}
                <span className="font-bold text-slate-800">{Math.round(response.timings.embedding_ms)}ms</span>
              </div>
            )}
            {response.timings.retrieval_ms && (
              <div className="rounded-md bg-slate-50 px-2.5 py-1.5 border border-slate-200">
                <span className="text-slate-400">Qdrant Search:</span>{' '}
                <span className="font-bold text-slate-800">{Math.round(response.timings.retrieval_ms)}ms</span>
              </div>
            )}
            {response.timings.total_ms && (
              <div className="rounded-md bg-indigo-50 px-2.5 py-1.5 border border-indigo-200 text-indigo-700">
                <span className="text-indigo-400">Total:</span>{' '}
                <span className="font-bold">{Math.round(response.timings.total_ms)}ms</span>
              </div>
            )}
          </div>
        </div>

        {/* View mode toggle */}
        <div className="mt-4">
          <SegmentedControl
            label="View"
            value={viewMode}
            onChange={setViewMode}
            options={[
              { value: 'patent', label: 'By Patent' },
              { value: 'feature', label: 'By Feature' },
            ]}
          />
        </div>

        {/* Feature quick selector pills (feature view only) */}
        {viewMode === 'feature' && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-slate-500">Filter feature:</span>
            <button
              type="button"
              onClick={() => setSelectedFeatureId('all')}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                selectedFeatureId === 'all'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              All Features ({results.length})
            </button>
            {results.map((r) => (
              <button
                key={r.feature_id}
                type="button"
                onClick={() => setSelectedFeatureId(r.feature_id)}
                className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                  selectedFeatureId === r.feature_id
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                Feature {r.feature_id} ({r.chunks?.length || 0})
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Results */}
      {viewMode === 'patent' ? (
        <KeyFeaturePatentTable results={results} />
      ) : (
        <div className="space-y-4">
          {displayedResults.map((result: FeatureSearchResult) => (
            <KeyFeatureCard
              key={result.feature_id}
              result={result}
              defaultExpanded={true}
            />
          ))}
        </div>
      )}
    </div>
  )
}

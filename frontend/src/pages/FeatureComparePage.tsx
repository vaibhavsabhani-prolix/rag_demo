import { Alert, EmptyState, ScaleIcon } from '@/components/ui'
import { FeatureCompareDashboard } from '@/features/feature-compare/FeatureCompareDashboard'
import { FeatureCompareForm } from '@/features/feature-compare/FeatureCompareForm'
import { FeatureCompareProgress } from '@/features/feature-compare/FeatureCompareProgress'
import { useSeries } from '@/features/compare/series'
import { useKeyFeatureCompare } from '@/hooks/useKeyFeatureCompare'

export function FeatureComparePage() {
  const compare = useKeyFeatureCompare()
  const { status, extractedFeatures, runs, collections, error, start, stop, isPending } = compare

  const activeCollections = collections.length > 0 ? collections : ['512', '2048', '4096']
  const series = useSeries(activeCollections)

  const hasAnyResults = Object.values(runs).some((r) => r.status === 'success' && r.response)

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Feature-Wise Collection Comparison
        </h1>
        <p className="text-slate-500 dark:text-white/70 max-w-3xl">
          Extract 5–15 key technical features from your invention problem and disclosure using an LLM, then search
          and rerank candidates across collections (e.g. 512, 2048, 4096 chunk sizes). Compare which patents match
          each feature, best scores, text passages, and cross-collection consensus.
        </p>
      </div>

      {/* Input Form */}
      <FeatureCompareForm
        onSubmit={start}
        onStop={stop}
        isPending={isPending}
        extractedFeatures={extractedFeatures}
        onClearFeatures={compare.clearFeatures}
      />

      {/* Idle State */}
      {status === 'idle' ? (
        <EmptyState
          icon={<ScaleIcon className="size-10" />}
          title="Start a feature-wise comparison"
          description="Enter your invention details and select collections to compare. Features are extracted once by the LLM, then searched and reranked across each collection independently."
        />
      ) : (
        <>
          {/* Progress Box (Rendered strictly once during comparison lifecycle) */}
          <FeatureCompareProgress state={compare} series={series} />

          {/* Error Alert */}
          {status === 'error' && error && (
            <Alert tone="danger" title="Comparison failed">
              {error}
            </Alert>
          )}

          {/* Feature Compare Dashboard */}
          {hasAnyResults && (
            <FeatureCompareDashboard
              features={extractedFeatures}
              runs={runs}
              series={series}
              totalElapsedMs={compare.elapsedMs}
            />
          )}
        </>
      )}
    </div>
  )
}


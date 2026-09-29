import { Alert, ChartIcon, EmptyState } from '@/components/ui'
import { CompareDashboard } from '@/features/compare/CompareDashboard'
import { CompareForm } from '@/features/compare/CompareForm'
import { CompareProgress } from '@/features/compare/CompareProgress'
import { collectionMetrics } from '@/features/compare/metrics'
import { useSeries } from '@/features/compare/series'
import { useAppSelector } from '@/store'
import { selectCompare } from '@/store/selectors'

export function ComparePage() {
  const state = useAppSelector(selectCompare)
  const series = useSeries(state.collections)
  const metrics = state.collections.map((name) => collectionMetrics(name, state.runs[name]))

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Compare collections</h1>
        <p className="text-slate-500">
          Run one query against several collections (for example different chunk sizes) and compare timing, how many
          patents each step keeps, and the results.
        </p>
      </div>

      <CompareForm />

      {state.status === 'idle' ? (
        <EmptyState
          icon={<ChartIcon className="size-10" />}
          title="Start a comparison"
          description="The query is understood once, then steps 2–7 run on each selected collection in turn. Charts fill in as each collection finishes."
        />
      ) : (
        <>
          {state.query && (
            <p className="text-sm text-slate-500">
              Comparing <span className="font-medium text-slate-900">“{state.query}”</span>
            </p>
          )}
          <CompareProgress state={state} series={series} />
          {state.status === 'error' && (
            <Alert tone="danger" title="Comparison stopped">
              {state.error}
            </Alert>
          )}
          <CompareDashboard state={state} metrics={metrics} series={series} />
        </>
      )}
    </div>
  )
}

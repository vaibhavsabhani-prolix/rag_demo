import { Badge, CheckIcon, CloseIcon, Spinner } from '@/components/ui'
import type { KeyFeatureCompareState } from '@/hooks/useKeyFeatureCompare'
import type { Series } from '@/features/compare/charts'

interface FeatureCompareProgressProps {
  state: KeyFeatureCompareState
  series: Series[]
}

export function FeatureCompareProgress({ state, series }: FeatureCompareProgressProps) {
  const { currentStep, stepName, extractedFeatures, collections, runs, isPending } = {
    ...state,
    isPending: state.status === 'running',
  }

  const seriesByCollection = Object.fromEntries(series.map((s) => [s.name, s]))

  return (
    <div className="rounded-xl border border-indigo-200/60 dark:border-white/10 bg-indigo-50/40 dark:bg-surface p-5 shadow-xs space-y-4">
      {/* Overall Header & Step */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {isPending ? (
            <Spinner className="size-5 text-indigo-600 dark:text-white" />
          ) : (
            <div className="flex size-6 items-center justify-center rounded-full bg-emerald-600 text-white">
              <CheckIcon className="size-4" />
            </div>
          )}
          <div>
            <p className="text-sm font-semibold text-slate-900 dark:text-white">
              {stepName || (state.status === 'success' ? 'Feature comparison completed' : 'Feature Comparison')}
            </p>
            <p className="text-xs text-slate-600 dark:text-white/80">
              {currentStep === 1 && 'Extracting key technical features with LLM (once for all collections)…'}
              {currentStep === 2 && 'Generating high-dimensional embeddings for all extracted features…'}
              {currentStep === 3 && 'Searching and reranking candidate chunks per collection…'}
              {state.status === 'success' && `Successfully compared across ${collections.length} collections.`}
            </p>
          </div>
        </div>

        {state.elapsedMs != null && (
          <Badge tone="brand" className="font-mono text-xs dark:text-white">
            {(state.elapsedMs / 1000).toFixed(1)}s total
          </Badge>
        )}
      </div>

      {/* Collection-by-Collection Progress Grid */}
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 md:grid-cols-3">
        {collections.map((name) => {
          const run = runs[name]
          const color = seriesByCollection[name]?.color || '#6366f1'
          const status = run?.status ?? 'idle'

          return (
            <div
              key={name}
              className={`flex items-center justify-between rounded-lg border p-3 text-xs transition-colors ${
                status === 'running'
                  ? 'border-indigo-500 bg-surface ring-2 ring-indigo-500/30 dark:border-indigo-400 dark:ring-indigo-400/40 shadow-xs'
                  : status === 'success'
                  ? 'border-slate-200 bg-surface dark:border-white/10'
                  : status === 'error'
                  ? 'border-rose-200 bg-rose-50/50 dark:border-rose-500/30'
                  : 'border-slate-200/80 bg-surface/50 dark:border-white/10 opacity-75'
              }`}
            >
              <div className="flex items-center gap-2.5 min-w-0">
                <span
                  className="size-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: color }}
                  aria-hidden
                />
                <div className="truncate">
                  <span className="font-semibold text-slate-900 dark:text-white">{name}</span>
                  <div className="text-2xs text-slate-500 dark:text-white/70 truncate">
                    {status === 'running' && (run?.stepName || 'Searching & Reranking…')}
                    {status === 'success' && `${run?.response?.total_features ?? 0} features matched`}
                    {status === 'error' && (run?.error || 'Failed')}
                    {status === 'idle' && 'Queued…'}
                  </div>
                </div>
              </div>

              <div className="shrink-0 ml-2">
                {status === 'running' && <Spinner className="size-3.5 text-indigo-600 dark:text-white" />}
                {status === 'success' && (
                  <span className="font-mono font-bold text-emerald-700 dark:text-emerald-400">
                    {run?.elapsedMs ? `${Math.round(run.elapsedMs)}ms` : 'Done'}
                  </span>
                )}
                {status === 'error' && <CloseIcon className="size-4 text-rose-500" />}
                {status === 'idle' && <span className="text-slate-400 dark:text-white/60">Waiting</span>}
              </div>
            </div>
          )
        })}
      </div>

      {/* Extracted Features Preview */}
      {extractedFeatures.length > 0 && (
        <div className="border-t border-indigo-200/60 dark:border-white/10 pt-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-2xs font-semibold uppercase tracking-wider text-indigo-900 dark:text-white font-bold">
              Extracted Technical Features ({extractedFeatures.length})
            </span>
            <span className="text-2xs text-slate-500 dark:text-white/70 font-mono">Shared across all collections</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-56 overflow-y-auto pr-1">
            {extractedFeatures.map((f) => (
              <div
                key={f.feature_id}
                className="flex items-start gap-2.5 rounded-lg bg-surface px-3 py-2 text-xs text-slate-800 dark:text-white shadow-2xs border border-indigo-200/70 dark:border-white/15"
              >
                <span className="font-mono font-bold text-indigo-600 dark:text-indigo-300 shrink-0 mt-0.5">
                  [{f.feature_id}]
                </span>
                <span className="text-slate-800 dark:text-white font-normal leading-relaxed break-words flex-1">
                  {f.feature}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

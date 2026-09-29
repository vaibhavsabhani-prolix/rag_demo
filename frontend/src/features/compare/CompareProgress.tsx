import clsx from 'clsx'
import { Badge } from '@/components/ui'
import { formatMs } from '@/lib/format'
import { PHASES } from '@/schemas/pipeline'
import type { CompareState } from '@/store/compareSlice'
import type { SearchRun } from '@/store/searchSlice'
import type { Series } from './charts'
import { COLLECTION_PHASES } from './metrics'

/** Where every collection is: queued, which step it's on, done or failed. */
export function CompareProgress({ state, series }: { state: CompareState; series: Series[] }) {
  const parse = Object.values(state.runs)[0]?.phases[1]

  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className="font-medium text-slate-700">Progress</span>
        <span className="text-slate-500">
          {state.status === 'running' && 'Collections run one at a time so their timings are comparable…'}
          {state.status === 'success' && `All done in ${formatMs(state.totalMs)}`}
          {state.status === 'error' && 'Stopped'}
        </span>
      </div>

      <p className="text-sm text-slate-500">
        Query understanding (shared, runs once):{' '}
        {parse ? (
          <span className="text-slate-700">
            {formatMs(parse.elapsedMs)}
            {state.cacheHit && ' · cache hit'}
          </span>
        ) : state.status === 'running' ? (
          'running…'
        ) : (
          'not run'
        )}
      </p>

      <ul className="space-y-2">
        {series.map((s) => {
          const run = state.runs[s.name]
          if (!run) return null
          return (
            <li
              key={s.name}
              className="grid grid-cols-[6.5rem_minmax(0,1fr)] items-center gap-3 sm:grid-cols-[9rem_minmax(0,1fr)_12rem]"
            >
              <span className="flex items-center gap-2 text-sm font-medium text-slate-700">
                <span className="size-2.5 shrink-0 rounded-sm" style={{ backgroundColor: s.color }} aria-hidden />
                <span className="truncate">{s.name}</span>
              </span>
              <StepBar run={run} />
              <span className="col-span-2 text-sm text-slate-500 sm:col-span-1 sm:text-right">
                <RunStatus run={run} />
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function StepBar({ run }: { run: SearchRun }) {
  const next = COLLECTION_PHASES.find((id) => !run.phases[id])
  return (
    <div className="grid grid-cols-6 gap-0.5" aria-hidden>
      {COLLECTION_PHASES.map((id) => (
        <span
          key={id}
          title={PHASES.find((p) => p.id === id)?.title}
          className={clsx(
            'h-1.5 rounded-full',
            run.phases[id]
              ? 'bg-indigo-500'
              : id === next && run.status === 'running'
                ? 'animate-pulse bg-indigo-300'
                : id === next && run.status === 'error'
                  ? 'bg-rose-400'
                  : 'bg-slate-100',
          )}
        />
      ))}
    </div>
  )
}

function RunStatus({ run }: { run: SearchRun }) {
  switch (run.status) {
    case 'idle':
      return <Badge>Queued</Badge>
    case 'running': {
      const next = PHASES.find((p) => p.id !== 1 && !run.phases[p.id])
      return <>Running · {next?.short ?? 'finishing'}…</>
    }
    case 'success':
      return <>Done in {formatMs(run.totalMs)}</>
    case 'error':
      return (
        <span className="text-rose-600" title={run.error}>
          Failed
        </span>
      )
  }
}

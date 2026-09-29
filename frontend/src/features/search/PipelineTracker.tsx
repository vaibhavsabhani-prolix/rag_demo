import clsx from 'clsx'
import { CheckIcon, CloseIcon, Spinner } from '@/components/ui'
import { formatMs } from '@/lib/format'
import { PHASES, type PhaseNumber } from '@/schemas/pipeline'
import type { SearchRun } from '@/store/searchSlice'

type StepState = 'done' | 'running' | 'failed' | 'pending'

/** Horizontal 7-step tracker that fills in as phase results stream in. */
export function PipelineTracker({ run }: { run: SearchRun }) {
  const { status, phases, totalMs, cacheHit, collection } = run
  const nextPhase = PHASES.find((p) => !phases[p.id])?.id

  const stepState = (id: PhaseNumber): StepState => {
    if (phases[id]) return 'done'
    if (id !== nextPhase) return 'pending'
    if (status === 'running') return 'running'
    if (status === 'error') return 'failed'
    return 'pending'
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className="font-medium text-slate-700">
          Pipeline
          {collection && <span className="font-normal text-slate-500"> · collection {collection}</span>}
        </span>
        <span className="text-slate-500">
          {status === 'running' && nextPhase && `Running step ${nextPhase} of 7…`}
          {status === 'success' && `Completed in ${formatMs(totalMs)}${cacheHit ? ' · query cache hit' : ''}`}
          {status === 'error' && 'Stopped with an error'}
        </span>
      </div>

      <ol className="grid grid-cols-4 gap-2 sm:grid-cols-7">
        {PHASES.map((phase) => {
          const state = stepState(phase.id)
          return (
            <li key={phase.id} className="min-w-0">
              <div
                className={clsx(
                  'mb-2 h-1.5 rounded-full',
                  state === 'done' && 'bg-indigo-500',
                  state === 'running' && 'animate-pulse bg-indigo-300',
                  state === 'failed' && 'bg-rose-400',
                  state === 'pending' && 'bg-slate-100',
                )}
              />
              <div className="flex items-center gap-1.5">
                <StepIcon state={state} index={phase.id} />
                <span
                  className={clsx(
                    'truncate text-xs font-medium',
                    state === 'pending' ? 'text-slate-400' : 'text-slate-700',
                  )}
                  title={phase.title}
                >
                  {phase.short}
                </span>
              </div>
              <div className="mt-0.5 pl-6 text-xs text-slate-400 tabular-nums">
                {phases[phase.id] ? formatMs(phases[phase.id]!.elapsedMs) : ' '}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

function StepIcon({ state, index }: { state: StepState; index: number }) {
  const base = 'flex size-4.5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold'
  switch (state) {
    case 'done':
      return (
        <span className={clsx(base, 'bg-indigo-600 text-white')}>
          <CheckIcon className="size-3" strokeWidth={3} />
        </span>
      )
    case 'running':
      return <Spinner className="size-4.5 text-indigo-500" />
    case 'failed':
      return (
        <span className={clsx(base, 'bg-rose-500 text-white')}>
          <CloseIcon className="size-3" strokeWidth={3} />
        </span>
      )
    default:
      return <span className={clsx(base, 'bg-slate-100 text-slate-400')}>{index}</span>
  }
}

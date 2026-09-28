import type { ReactNode } from 'react'
import { Badge, Collapsible, EmptyState, JsonView, Spinner } from '@/components/ui'
import { formatMs } from '@/lib/format'
import { PHASES, type PhaseNumber } from '@/schemas/pipeline'
import type { SearchRun } from '@/store/searchSlice'
import { EvidencePhase } from './phases/EvidencePhase'
import { FilterPhase } from './phases/FilterPhase'
import { QueryPhase } from './phases/QueryPhase'
import { RerankPhase } from './phases/RerankPhase'
import { RetrievalPhase } from './phases/RetrievalPhase'
import { ScoringPhase } from './phases/ScoringPhase'
import { VerificationPhase } from './phases/VerificationPhase'

/** Step-by-step view of every pipeline phase's output. */
export function PipelineView({ run }: { run: SearchRun }) {
  const { status, phases, cacheHit } = run

  if (status === 'idle') {
    return <EmptyState title="No pipeline run yet" description="Run a search to inspect what each of the 7 steps produced." />
  }

  const renderPhase = (id: PhaseNumber): ReactNode => {
    switch (id) {
      case 1:
        return phases[1] && <QueryPhase data={phases[1].data} cacheHit={cacheHit} />
      case 2:
        return phases[2] && <RetrievalPhase data={phases[2].data} />
      case 3:
        return phases[3] && <FilterPhase data={phases[3].data} />
      case 4:
        return phases[4] && <EvidencePhase data={phases[4].data} />
      case 5:
        return phases[5] && <VerificationPhase data={phases[5].data} />
      case 6:
        return phases[6] && <RerankPhase data={phases[6].data} />
      case 7:
        return phases[7] && <ScoringPhase data={phases[7].data} />
    }
  }

  const nextPhase = PHASES.find((p) => !phases[p.id])?.id

  return (
    <div className="space-y-2">
      {PHASES.map(({ id, title }) => {
        const entry = phases[id]
        const running = !entry && status === 'running' && id === nextPhase
        return (
          <Collapsible
            key={id}
            disabled={!entry}
            title={`${id}. ${title}`}
            aside={
              entry ? (
                <Badge>{formatMs(entry.elapsedMs)}</Badge>
              ) : running ? (
                <Spinner className="size-4 text-indigo-500" />
              ) : (
                <span className="text-xs text-slate-400">{status === 'error' && id === nextPhase ? 'failed' : 'pending'}</span>
              )
            }
          >
            <div className="space-y-4">
              {renderPhase(id)}
              <JsonView data={entry?.data} />
            </div>
          </Collapsible>
        )
      })}
    </div>
  )
}

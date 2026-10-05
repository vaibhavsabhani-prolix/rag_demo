import { useEffect } from 'react'
import { Alert, Tabs } from '@/components/ui'
import { PipelineView } from '@/features/pipeline/PipelineView'
import { PatentDetailDrawer } from '@/features/results/PatentDetailDrawer'
import { ResultsView } from '@/features/results/ResultsView'
import { PipelineTracker } from '@/features/search/PipelineTracker'
import { PHASES } from '@/schemas/pipeline'
import { useAppDispatch, useAppSelector } from '@/store'
import type { SearchRun } from '@/store/searchSlice'
import { patentSelected, tabChanged, type ResultTab } from '@/store/uiSlice'

/** Everything shown for one pipeline run: progress, results and per-phase details. */
export function SearchRunView({ run }: { run: SearchRun }) {
  const dispatch = useAppDispatch()
  const activeTab = useAppSelector((s) => s.ui.activeTab)
  const final = run.phases[6]
  const completedPhases = PHASES.filter((p) => run.phases[p.id]).length

  // Close the patent drawer when leaving this run (e.g. switching pages).
  useEffect(() => () => void dispatch(patentSelected(null)), [dispatch])

  return (
    <div className="space-y-6">
      {run.status !== 'idle' && <PipelineTracker run={run} />}
      {run.status === 'error' && (
        <Alert tone="danger" title="Search failed">
          {run.error}
        </Alert>
      )}

      <div className="space-y-4">
        <Tabs<ResultTab>
          value={activeTab}
          onChange={(tab) => dispatch(tabChanged(tab))}
          items={[
            { id: 'results', label: 'Results', count: final ? final.data.results.length : undefined },
            { id: 'pipeline', label: 'Pipeline', count: run.status === 'idle' ? undefined : completedPhases },
          ]}
        />
        {activeTab === 'results' ? <ResultsView run={run} /> : <PipelineView run={run} />}
      </div>

      <PatentDetailDrawer run={run} />
    </div>
  )
}

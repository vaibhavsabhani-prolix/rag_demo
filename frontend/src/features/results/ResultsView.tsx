import { Alert, EmptyState, SearchIcon, Spinner } from '@/components/ui'
import { useAppDispatch } from '@/store'
import type { SearchRun } from '@/store/searchSlice'
import { patentSelected } from '@/store/uiSlice'
import { ResultCard } from './ResultCard'

export function ResultsView({ run }: { run: SearchRun }) {
  const dispatch = useAppDispatch()
  const { status } = run
  const final = run.phases[6]

  if (!final) {
    if (status === 'running') {
      return (
        <EmptyState
          icon={<Spinner className="size-10" />}
          title="Searching patents…"
          description="Ranked results appear here when the final scoring step finishes. Open the Pipeline tab to follow each step live."
        />
      )
    }
    if (status === 'error') {
      return <EmptyState title="No results" description="The search stopped before scoring finished." />
    }
    return (
      <EmptyState
        icon={<SearchIcon className="size-10" />}
        title="Start with a search"
        description="Describe an invention in plain language. The pipeline extracts concepts and relationships, retrieves candidates, verifies evidence and ranks the best matches."
      />
    )
  }

  const { results, total_candidates_evaluated, rejected_count, threshold_used } = final.data
  // A metadata-only query has nothing semantic to score or threshold (see app/scoring/scorer.py).
  const isMetadataOnly = results.length > 0 && results.every((r) => r.final_score === null)

  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-500">
        {isMetadataOnly ? (
          <>
            <strong className="text-slate-900">{results.length}</strong> patent{results.length === 1 ? '' : 's'}{' '}
            matched this metadata filter out of {total_candidates_evaluated} evaluated (not scored).
          </>
        ) : (
          <>
            <strong className="text-slate-900">{results.length}</strong> patent{results.length === 1 ? '' : 's'}{' '}
            scored ≥ {threshold_used} out of {total_candidates_evaluated} evaluated ({rejected_count} below
            threshold).
          </>
        )}
      </p>

      {results.length === 0 ? (
        <Alert tone="warning" title="No qualifying patents">
          No candidate reached the final score threshold of {threshold_used} / 10. Try a broader query.
        </Alert>
      ) : (
        <div className="space-y-3">
          {results.map((patent, i) => (
            <ResultCard
              key={patent.patent_id}
              patent={patent}
              rank={i + 1}
              onOpen={(id) => dispatch(patentSelected(id))}
            />
          ))}
        </div>
      )}
    </div>
  )
}

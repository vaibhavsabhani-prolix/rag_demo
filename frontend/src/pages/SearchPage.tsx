import { Link } from 'react-router'
import { SearchRunView } from '@/features/run/SearchRunView'
import { SearchForm } from '@/features/search/SearchForm'
import { useAppSelector } from '@/store'
import { selectSearch } from '@/store/selectors'

export function SearchPage() {
  const run = useAppSelector(selectSearch)

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Find prior art by meaning</h1>
        <p className="text-slate-500">
          Semantic patent search that understands concepts and relationships, verifies evidence, and ranks the
          strongest matches.
        </p>
      </div>

      <SearchForm />

      {run.searchId && run.status !== 'running' && (
        <p className="text-sm text-slate-500">
          Saved to{' '}
          <Link to={`/history/${run.searchId}`} className="font-medium text-indigo-600 hover:underline">
            history
          </Link>
          .
        </p>
      )}

      <SearchRunView run={run} />
    </div>
  )
}

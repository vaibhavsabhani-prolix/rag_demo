import { Link, useNavigate, useParams } from 'react-router'
import { Alert, ArrowLeftIcon, Button, ConfirmButton, EmptyState, SearchIcon, Spinner } from '@/components/ui'
import { HistoryStatusBadge } from '@/features/history/HistoryStatusBadge'
import { SearchRunView } from '@/features/run/SearchRunView'
import { useDeleteHistoryItem, useHistoryDetail } from '@/hooks/queries'
import { ApiError } from '@/lib/api'
import { formatDate, formatMs } from '@/lib/format'

export function HistoryDetailPage() {
  const id = Number(useParams().id)
  const navigate = useNavigate()
  const { data, isPending, error } = useHistoryDetail(id)
  const remove = useDeleteHistoryItem()

  const backLink = (
    <Link to="/history" className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-900">
      <ArrowLeftIcon className="size-4" />
      All searches
    </Link>
  )

  if (!Number.isInteger(id) || id <= 0 || (error instanceof ApiError && error.status === 404)) {
    return (
      <div className="space-y-6">
        {backLink}
        <EmptyState title="Search not found" description="It may have been deleted." />
      </div>
    )
  }
  if (error) return <Alert tone="danger" title="Could not load this search">{error.message}</Alert>
  if (isPending) {
    return (
      <div className="flex justify-center py-24 text-slate-400">
        <Spinner className="size-8" />
      </div>
    )
  }

  const { item, run } = data
  const searchAgainParams = new URLSearchParams({ q: item.query })
  if (item.collection) searchAgainParams.set('collection', item.collection)

  return (
    <div className="space-y-6">
      {backLink}

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-bold tracking-tight break-words text-slate-900">{item.query}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-slate-500">
            <HistoryStatusBadge status={item.status} />
            <span>{formatDate(item.created_at)}</span>
            {item.collection && <span>· collection {item.collection}</span>}
            {item.total_ms !== null && <span>· {formatMs(item.total_ms)}</span>}
            <span>· cache {item.use_cache ? (item.cache_hit ? 'hit' : 'miss') : 'off'}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={() => navigate(`/?${searchAgainParams}`)}>
            <SearchIcon className="size-4" />
            Search again
          </Button>
          <ConfirmButton
            variant="danger"
            size="sm"
            confirmLabel="Delete?"
            loading={remove.isPending}
            onConfirm={() => remove.mutate(item.id, { onSuccess: () => navigate('/history') })}
          >
            Delete
          </ConfirmButton>
        </div>
      </div>

      <SearchRunView run={run} />
    </div>
  )
}

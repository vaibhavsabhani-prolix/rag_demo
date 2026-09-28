import { zodResolver } from '@hookform/resolvers/zod'
import { useForm, useWatch } from 'react-hook-form'
import { useSearchParams } from 'react-router'
import { Alert, ConfirmButton, EmptyState, HistoryIcon, Pagination, SearchIcon, Spinner, TextField } from '@/components/ui'
import { HistoryCard } from '@/features/history/HistoryCard'
import { useClearHistory, useHistoryList } from '@/hooks/queries'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { historyFilterSchema } from '@/schemas/history'

const PAGE_SIZE = 20

export function HistoryPage() {
  // Page number lives in the URL so back/forward and reloads keep it.
  const [params, setParams] = useSearchParams()
  const page = Math.max(0, Number(params.get('page') ?? 0) || 0)

  const {
    register,
    control,
    formState: { errors },
  } = useForm({ resolver: zodResolver(historyFilterSchema), defaultValues: { q: '' }, mode: 'onChange' })
  const filter = useDebouncedValue(useWatch({ control, name: 'q' }).trim())
  const validFilter = errors.q ? '' : filter

  const { data, isPending, isFetching, error } = useHistoryList({
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
    q: validFilter || undefined,
  })
  const clear = useClearHistory()

  const setPage = (next: number) => setParams(next > 0 ? { page: String(next) } : {})

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Search history</h1>
          <p className="mt-1 text-slate-500">Every search is saved with its full results. Open one to review it.</p>
        </div>
        {!!data?.total && (
          <ConfirmButton
            variant="danger"
            size="sm"
            confirmLabel="Delete all history?"
            loading={clear.isPending}
            onConfirm={() => clear.mutate(undefined, { onSuccess: () => setPage(0) })}
          >
            Clear history
          </ConfirmButton>
        )}
      </div>

      <TextField
        {...register('q', { onChange: () => setPage(0) })}
        aria-label="Filter history"
        placeholder="Filter by query text…"
        leading={<SearchIcon className="size-5" />}
        error={errors.q?.message}
      />

      {error && <Alert tone="danger" title="Could not load history">{error.message}</Alert>}
      {clear.error && <Alert tone="danger">{clear.error.message}</Alert>}

      {isPending ? (
        <div className="flex justify-center py-16 text-slate-400">
          <Spinner className="size-8" />
        </div>
      ) : data && data.items.length === 0 ? (
        <EmptyState
          icon={<HistoryIcon className="size-10" />}
          title={validFilter ? 'No matching searches' : 'No searches yet'}
          description={validFilter ? 'Try a different filter.' : 'Searches you run are saved here automatically.'}
        />
      ) : (
        data && (
          <div className={isFetching ? 'space-y-3 opacity-60 transition-opacity' : 'space-y-3'}>
            {data.items.map((item) => (
              <HistoryCard key={item.id} item={item} />
            ))}
          </div>
        )
      )}

      {data && <Pagination page={page} pageSize={PAGE_SIZE} total={data.total} onChange={setPage} />}
    </div>
  )
}

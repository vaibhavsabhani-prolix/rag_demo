import { Link } from 'react-router'
import { Badge, Card, ConfirmButton, TrashIcon } from '@/components/ui'
import { useDeleteHistoryItem } from '@/hooks/queries'
import { formatDate, formatMs, formatRelative } from '@/lib/format'
import { finalScoreTone } from '@/lib/tone'
import type { HistoryItem } from '@/schemas/history'
import { HistoryStatusBadge } from './HistoryStatusBadge'

/** One saved search in the history list. The whole card links to its detail page. */
export function HistoryCard({ item }: { item: HistoryItem }) {
  const remove = useDeleteHistoryItem()

  return (
    <Card className="relative p-5 transition hover:border-indigo-300 hover:shadow-md">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <Link
            to={`/history/${item.id}`}
            className="font-medium text-slate-900 after:absolute after:inset-0 hover:text-indigo-700"
          >
            {item.query}
          </Link>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-slate-500">
            <time dateTime={item.created_at} title={formatDate(item.created_at)}>
              {formatRelative(item.created_at)}
            </time>
            {item.collection && <span>· {item.collection}</span>}
            {item.total_ms !== null && <span>· {formatMs(item.total_ms)}</span>}
            {item.cache_hit && <span>· cache hit</span>}
          </div>
        </div>

        <div className="relative flex shrink-0 items-center gap-2">
          <HistoryStatusBadge status={item.status} />
          <ConfirmButton
            variant="ghost"
            size="sm"
            aria-label="Delete search"
            confirmLabel="Delete?"
            loading={remove.isPending}
            onConfirm={() => remove.mutate(item.id)}
          >
            <TrashIcon className="size-4" />
          </ConfirmButton>
        </div>
      </div>

      {item.status === 'success' && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-sm text-slate-600">
            <strong>{item.result_count}</strong> result{item.result_count === 1 ? '' : 's'}
          </span>
          {item.top_results.map((r) => (
            <Badge key={r.patent_id} tone={finalScoreTone(r.final_score)} className="font-mono">
              {r.patent_id}
              {r.final_score !== null && ` · ${r.final_score.toFixed(2)}`}
            </Badge>
          ))}
        </div>
      )}
      {item.error && <p className="mt-3 line-clamp-1 text-sm text-rose-600">{item.error}</p>}
    </Card>
  )
}

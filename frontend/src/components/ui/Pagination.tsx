import { Button } from './Button'

interface PaginationProps {
  /** Zero-based page index. */
  page: number
  pageSize: number
  total: number
  onChange: (page: number) => void
}

export function Pagination({ page, pageSize, total, onChange }: PaginationProps) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  if (pageCount <= 1) return null

  const from = page * pageSize + 1
  const to = Math.min(total, (page + 1) * pageSize)

  return (
    <nav className="flex items-center justify-between gap-3 text-sm" aria-label="Pagination">
      <span className="text-slate-500">
        {from}–{to} of {total}
      </span>
      <div className="flex gap-2">
        <Button variant="secondary" size="sm" disabled={page === 0} onClick={() => onChange(page - 1)}>
          Previous
        </Button>
        <Button variant="secondary" size="sm" disabled={page >= pageCount - 1} onClick={() => onChange(page + 1)}>
          Next
        </Button>
      </div>
    </nav>
  )
}

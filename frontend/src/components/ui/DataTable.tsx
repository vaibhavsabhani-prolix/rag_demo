import clsx from 'clsx'
import type { ReactNode } from 'react'

export interface Column<T> {
  header: string
  render: (row: T, index: number) => ReactNode
  align?: 'left' | 'right'
  className?: string
}

interface DataTableProps<T> {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T, index: number) => string
  emptyText?: string
}

export function DataTable<T>({ columns, rows, rowKey, emptyText = 'No rows.' }: DataTableProps<T>) {
  if (rows.length === 0) return <p className="text-sm text-slate-500">{emptyText}</p>

  return (
    <div className="overflow-x-auto rounded-lg ring-1 ring-slate-200">
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50">
          <tr>
            {columns.map((col) => (
              <th
                key={col.header}
                scope="col"
                className={clsx(
                  'px-3 py-2 text-xs font-semibold tracking-wide whitespace-nowrap text-slate-500 uppercase',
                  col.align === 'right' ? 'text-right' : 'text-left',
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-surface">
          {rows.map((row, i) => (
            <tr key={rowKey(row, i)} className="hover:bg-slate-50">
              {columns.map((col) => (
                <td
                  key={col.header}
                  className={clsx(
                    'px-3 py-2 align-top text-slate-700',
                    col.align === 'right' && 'text-right tabular-nums',
                    col.className,
                  )}
                >
                  {col.render(row, i)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

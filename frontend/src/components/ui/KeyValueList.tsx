import type { ReactNode } from 'react'

export interface KeyValueItem {
  label: string
  value: ReactNode
}

/** Two-column list of label → value pairs. */
export function KeyValueList({ items }: { items: KeyValueItem[] }) {
  return (
    <dl className="divide-y divide-slate-100">
      {items.map(({ label, value }) => (
        <div key={label} className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-4 py-2 text-sm">
          <dt className="text-slate-500">{label}</dt>
          <dd className="font-medium break-words text-slate-900">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

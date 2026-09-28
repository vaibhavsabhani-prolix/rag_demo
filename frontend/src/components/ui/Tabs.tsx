import clsx from 'clsx'

export interface TabItem<T extends string> {
  id: T
  label: string
  count?: number
}

interface TabsProps<T extends string> {
  items: TabItem<T>[]
  value: T
  onChange: (id: T) => void
}

export function Tabs<T extends string>({ items, value, onChange }: TabsProps<T>) {
  return (
    <div role="tablist" className="inline-flex rounded-lg bg-slate-100 p-1">
      {items.map((item) => {
        const active = item.id === value
        return (
          <button
            key={item.id}
            role="tab"
            type="button"
            aria-selected={active}
            onClick={() => onChange(item.id)}
            className={clsx(
              'inline-flex items-center gap-2 rounded-md px-4 py-1.5 text-sm font-medium transition-colors',
              active ? 'bg-surface text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800',
            )}
          >
            {item.label}
            {item.count !== undefined && (
              <span
                className={clsx(
                  'rounded-full px-1.5 text-xs tabular-nums',
                  active ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-200 text-slate-600',
                )}
              >
                {item.count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

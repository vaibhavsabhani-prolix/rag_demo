import clsx from 'clsx'

interface SegmentedControlProps<T extends string> {
  label: string
  options: { value: T; label: string }[]
  value: T
  onChange: (value: T) => void
}

/** A row of mutually exclusive options (radio group styled as buttons). */
export function SegmentedControl<T extends string>({ label, options, value, onChange }: SegmentedControlProps<T>) {
  return (
    <div>
      <div className="mb-1.5 text-sm font-medium text-slate-700">{label}</div>
      <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg bg-slate-100 p-1">
        {options.map((option) => {
          const selected = option.value === value
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(option.value)}
              className={clsx(
                'rounded-md px-4 py-1.5 text-sm font-medium transition-colors',
                selected ? 'bg-surface text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800',
              )}
            >
              {option.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

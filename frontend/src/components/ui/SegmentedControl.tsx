import clsx from 'clsx'

interface SegmentedControlProps<T extends string> {
  label?: string
  options: { value: T; label: string }[]
  value: T
  onChange: (value: T) => void
  size?: 'sm' | 'md'
  className?: string
}

/** A row of mutually exclusive options (radio group styled as buttons). */
export function SegmentedControl<T extends string>({
  label,
  options,
  value,
  onChange,
  size = 'md',
  className,
}: SegmentedControlProps<T>) {
  return (
    <div className={className}>
      {label && (
        <div className="mb-1 text-xs font-semibold text-slate-700 dark:text-white">{label}</div>
      )}
      <div
        role="radiogroup"
        aria-label={label}
        className={clsx(
          'inline-flex items-center rounded-lg bg-slate-100 dark:bg-white/10 p-0.5 border border-slate-200/60 dark:border-white/10',
          size === 'sm' ? 'p-0.5' : 'p-1'
        )}
      >
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
                'rounded-md font-medium transition-all text-xs',
                size === 'sm' ? 'px-2.5 py-1 text-2xs' : 'px-3 py-1.5 text-xs',
                selected
                  ? 'bg-surface text-slate-900 dark:text-white shadow-2xs font-semibold ring-1 ring-slate-200/50 dark:ring-white/10'
                  : 'text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/50 dark:hover:bg-white/5'
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

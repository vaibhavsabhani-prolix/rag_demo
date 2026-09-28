import { useId } from 'react'

interface RangeFieldProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  hint?: string
  error?: string
  format?: (value: number) => string
}

/** A labelled slider showing its current value. */
export function RangeField({ label, value, onChange, min, max, step, hint, error, format = String }: RangeFieldProps) {
  const id = useId()
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium text-slate-700">
          {label}
        </label>
        <span className="font-mono text-sm font-semibold text-slate-900 tabular-nums">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-indigo-600"
      />
      {(error || hint) && <p className={error ? 'mt-1 text-sm text-rose-600' : 'mt-1 text-xs text-slate-500'}>{error ?? hint}</p>}
    </div>
  )
}

import clsx from 'clsx'
import type { Tone } from '@/lib/tone'

const fills: Record<Tone, string> = {
  neutral: 'bg-slate-400',
  brand: 'bg-indigo-500',
  success: 'bg-emerald-500',
  warning: 'bg-amber-500',
  danger: 'bg-rose-500',
  info: 'bg-sky-500',
}

interface ScoreBarProps {
  label: string
  value: number
  max?: number
  /** Text shown on the right; defaults to "value / max". */
  display?: string
  tone?: Tone
}

/** Labelled horizontal bar showing value out of max. */
export function ScoreBar({ label, value, max = 1, display, tone = 'brand' }: ScoreBarProps) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
        <span className="text-slate-600">{label}</span>
        <span className="font-medium text-slate-900 tabular-nums">{display ?? `${value.toFixed(2)} / ${max}`}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={clsx('h-full rounded-full transition-[width] duration-500', fills[tone])} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

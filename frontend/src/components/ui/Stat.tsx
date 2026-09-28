import clsx from 'clsx'
import type { ReactNode } from 'react'

interface StatProps {
  label: string
  value: ReactNode
  hint?: ReactNode
  className?: string
}

/** A single labelled metric. */
export function Stat({ label, value, hint, className }: StatProps) {
  return (
    <div className={clsx('rounded-lg bg-slate-50 px-4 py-3 ring-1 ring-slate-100', className)}>
      <div className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-900 tabular-nums">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  )
}

/** Responsive grid for a row of Stat tiles. */
export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 md:grid-cols-4">{children}</div>
}

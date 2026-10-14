import clsx from 'clsx'
import type { ReactNode } from 'react'

interface StatProps {
  label: ReactNode
  value: ReactNode
  hint?: ReactNode
  className?: string
}

/** A single labelled metric. */
export function Stat({ label, value, hint, className }: StatProps) {
  return (
    <div
      className={clsx(
        'rounded-lg bg-slate-50 dark:bg-white/5 px-4 py-3 ring-1 ring-slate-100 dark:ring-white/10',
        className
      )}
    >
      <div className="text-xs font-medium tracking-wide text-slate-500 dark:text-white/70 uppercase">
        {label}
      </div>
      <div className="mt-1 text-xl font-semibold text-slate-900 dark:text-white tabular-nums">
        {value}
      </div>
      {hint && <div className="mt-0.5 text-xs text-slate-500 dark:text-white/60">{hint}</div>}
    </div>
  )
}

/** Responsive grid for a row of Stat tiles. */
export function StatGrid({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={clsx('grid grid-cols-2 gap-3 md:grid-cols-4', className)}>{children}</div>
}

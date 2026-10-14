import clsx from 'clsx'
import type { ReactNode } from 'react'
import type { Tone } from '@/lib/tone'

const tones: Record<Tone, string> = {
  neutral: 'bg-slate-100 text-slate-700 ring-slate-200 dark:bg-white/10 dark:text-white dark:ring-white/20',
  brand: 'bg-indigo-50 text-indigo-700 ring-indigo-200 dark:bg-indigo-500/30 dark:text-white dark:ring-indigo-400/40',
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-200 dark:bg-emerald-500/30 dark:text-white dark:ring-emerald-400/40',
  warning: 'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-500/30 dark:text-white dark:ring-amber-400/40',
  danger: 'bg-rose-50 text-rose-700 ring-rose-200 dark:bg-rose-500/30 dark:text-white dark:ring-rose-400/40',
  info: 'bg-sky-50 text-sky-700 ring-sky-200 dark:bg-sky-500/30 dark:text-white dark:ring-sky-400/40',
}

interface BadgeProps {
  tone?: Tone
  className?: string
  children: ReactNode
}

export function Badge({ tone = 'neutral', className, children }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

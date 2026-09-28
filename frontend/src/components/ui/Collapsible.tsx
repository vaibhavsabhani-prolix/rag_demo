import clsx from 'clsx'
import { useState, type ReactNode } from 'react'
import { ChevronIcon } from './icons'

interface CollapsibleProps {
  title: ReactNode
  subtitle?: ReactNode
  /** Shown on the right of the header, e.g. a badge or timing. */
  aside?: ReactNode
  defaultOpen?: boolean
  disabled?: boolean
  children: ReactNode
}

/** A bordered section whose body can be expanded and collapsed. */
export function Collapsible({ title, subtitle, aside, defaultOpen = false, disabled, children }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen)
  const expanded = open && !disabled

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-surface">
      <button
        type="button"
        disabled={disabled}
        aria-expanded={expanded}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-5 py-4 text-left hover:bg-slate-50 disabled:cursor-default disabled:hover:bg-transparent"
      >
        <ChevronIcon className={clsx('size-4 shrink-0 text-slate-400 transition-transform', expanded && 'rotate-90')} />
        <div className="min-w-0 flex-1">
          <div className={clsx('font-medium', disabled ? 'text-slate-400' : 'text-slate-900')}>{title}</div>
          {subtitle && <div className="truncate text-sm text-slate-500">{subtitle}</div>}
        </div>
        {aside && <div className="shrink-0">{aside}</div>}
      </button>
      {expanded && <div className="border-t border-slate-100 px-5 py-4">{children}</div>}
    </div>
  )
}

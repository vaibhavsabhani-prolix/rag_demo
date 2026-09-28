import clsx from 'clsx'
import type { ReactNode } from 'react'

type AlertTone = 'info' | 'warning' | 'danger'

const tones: Record<AlertTone, string> = {
  info: 'bg-sky-50 text-sky-800 ring-sky-200',
  warning: 'bg-amber-50 text-amber-900 ring-amber-200',
  danger: 'bg-rose-50 text-rose-800 ring-rose-200',
}

interface AlertProps {
  tone?: AlertTone
  title?: ReactNode
  children?: ReactNode
  className?: string
}

export function Alert({ tone = 'info', title, children, className }: AlertProps) {
  return (
    <div role={tone === 'danger' ? 'alert' : 'status'} className={clsx('rounded-lg px-4 py-3 text-sm ring-1', tones[tone], className)}>
      {title && <div className="font-semibold">{title}</div>}
      {children && <div className={clsx('whitespace-pre-wrap', title && 'mt-1')}>{children}</div>}
    </div>
  )
}

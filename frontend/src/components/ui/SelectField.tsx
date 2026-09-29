import clsx from 'clsx'
import { useId, type ReactNode, type Ref, type SelectHTMLAttributes } from 'react'
import { ChevronIcon } from './icons'

interface SelectFieldProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  hint?: ReactNode
  error?: string
  ref?: Ref<HTMLSelectElement>
}

/** Native select styled like TextField. Pass <option> elements as children. */
export function SelectField({ label, hint, error, className, id, ref, children, ...props }: SelectFieldProps) {
  const generatedId = useId()
  const selectId = id ?? generatedId
  const messageId = `${selectId}-message`

  return (
    <div className={className}>
      {label && (
        <label htmlFor={selectId} className="mb-1.5 block text-sm font-medium text-slate-700">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          ref={ref}
          id={selectId}
          aria-invalid={!!error}
          aria-describedby={error || hint ? messageId : undefined}
          className={clsx(
            'block w-full appearance-none rounded-lg border bg-surface py-2.5 pr-9 pl-3.5 text-slate-900',
            'focus:outline-none focus:ring-4 disabled:cursor-not-allowed disabled:text-slate-400',
            error
              ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
              : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-100',
          )}
          {...props}
        >
          {children}
        </select>
        <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3 text-slate-400">
          <ChevronIcon className="size-4 rotate-90" />
        </span>
      </div>
      {(error || hint) && (
        <p id={messageId} className={clsx('mt-1.5 text-sm', error ? 'text-rose-600' : 'text-slate-500')}>
          {error ?? hint}
        </p>
      )}
    </div>
  )
}

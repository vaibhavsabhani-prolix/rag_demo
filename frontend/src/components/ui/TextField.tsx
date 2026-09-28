import clsx from 'clsx'
import { useId, type InputHTMLAttributes, type ReactNode, type Ref } from 'react'

interface TextFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  hint?: string
  error?: string
  /** Rendered inside the field on the left, e.g. an icon. */
  leading?: ReactNode
  ref?: Ref<HTMLInputElement>
}

export function TextField({ label, hint, error, leading, className, id, ref, ...props }: TextFieldProps) {
  const generatedId = useId()
  const inputId = id ?? generatedId
  const messageId = `${inputId}-message`

  return (
    <div className={className}>
      {label && (
        <label htmlFor={inputId} className="mb-1.5 block text-sm font-medium text-slate-700">
          {label}
        </label>
      )}
      <div className="relative">
        {leading && (
          <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5 text-slate-400">
            {leading}
          </span>
        )}
        <input
          ref={ref}
          id={inputId}
          aria-invalid={!!error}
          aria-describedby={error || hint ? messageId : undefined}
          className={clsx(
            'block w-full rounded-lg border bg-surface py-2.5 pr-3.5 text-slate-900 placeholder:text-slate-400',
            'focus:outline-none focus:ring-4',
            leading ? 'pl-10' : 'pl-3.5',
            error
              ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
              : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-100',
          )}
          {...props}
        />
      </div>
      {(error || hint) && (
        <p id={messageId} className={clsx('mt-1.5 text-sm', error ? 'text-rose-600' : 'text-slate-500')}>
          {error ?? hint}
        </p>
      )}
    </div>
  )
}

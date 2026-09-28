import { useId, type InputHTMLAttributes, type Ref } from 'react'

interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label: string
  ref?: Ref<HTMLInputElement>
}

export function Checkbox({ label, id, ref, ...props }: CheckboxProps) {
  const generatedId = useId()
  const inputId = id ?? generatedId
  return (
    <label htmlFor={inputId} className="inline-flex cursor-pointer items-center gap-2 text-sm text-slate-600 select-none">
      <input
        ref={ref}
        id={inputId}
        type="checkbox"
        className="size-4 rounded border-slate-300 accent-indigo-600"
        {...props}
      />
      {label}
    </label>
  )
}

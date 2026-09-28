import clsx from 'clsx'
import { useId } from 'react'
import { CheckIcon } from './icons'

interface ColorFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  presets?: { name: string; value: string }[]
  error?: string
}

/** Colour picker: a row of preset swatches plus a custom colour input. */
export function ColorField({ label, value, onChange, presets = [], error }: ColorFieldProps) {
  const id = useId()
  const isPreset = presets.some((p) => p.value.toLowerCase() === value.toLowerCase())

  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm font-medium text-slate-700">
        {label}
      </label>
      <div className="flex flex-wrap items-center gap-2">
        {presets.map((preset) => {
          const selected = preset.value.toLowerCase() === value.toLowerCase()
          return (
            <button
              key={preset.value}
              type="button"
              title={preset.name}
              aria-label={preset.name}
              aria-pressed={selected}
              onClick={() => onChange(preset.value)}
              className={clsx(
                'flex size-7 items-center justify-center rounded-full ring-offset-2 transition',
                selected ? 'ring-2 ring-slate-900' : 'hover:ring-2 hover:ring-slate-300',
              )}
              style={{ backgroundColor: preset.value }}
            >
              {selected && <CheckIcon className="size-4 text-white" strokeWidth={3} />}
            </button>
          )
        })}
        <label
          className={clsx(
            'relative flex h-7 cursor-pointer items-center gap-1.5 rounded-full border px-2 text-xs text-slate-600',
            !isPreset ? 'border-slate-900' : 'border-slate-300 hover:border-slate-400',
          )}
          title="Custom colour"
        >
          <span className="size-4 rounded-full ring-1 ring-black/10" style={{ backgroundColor: value }} />
          <span className="font-mono">{value.toLowerCase()}</span>
          <input
            id={id}
            type="color"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="absolute inset-0 cursor-pointer opacity-0"
          />
        </label>
      </div>
      {error && <p className="mt-1.5 text-sm text-rose-600">{error}</p>}
    </div>
  )
}

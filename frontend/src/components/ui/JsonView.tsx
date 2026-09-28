import { useState } from 'react'
import { Button } from './Button'

/** A toggle that reveals raw JSON for debugging. */
export function JsonView({ data, label = 'raw JSON' }: { data: unknown; label?: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <Button variant="ghost" size="sm" onClick={() => setOpen((v) => !v)}>
        {open ? `Hide ${label}` : `Show ${label}`}
      </Button>
      {open && (
        <pre className="mt-2 max-h-96 overflow-auto rounded-lg bg-slate-900 p-4 font-mono text-xs leading-relaxed text-slate-100">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}

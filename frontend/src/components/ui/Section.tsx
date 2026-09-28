import type { ReactNode } from 'react'

/** A small titled block used to group content inside cards and drawers. */
export function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{title}</h4>
        {aside}
      </div>
      {children}
    </section>
  )
}

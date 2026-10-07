import type { ReactNode } from 'react'
import type { RelationshipVerification, RequirementVerification } from '@/schemas/pipeline'

interface VerificationRowProps {
  label: ReactNode
  score: number
  chunkIds: number[]
}

function VerificationRow({ label, score, chunkIds }: VerificationRowProps) {
  return (
    <li className="py-2.5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1 text-sm text-slate-800">{label}</div>
        <span className="shrink-0 text-xs font-medium text-slate-600 tabular-nums">Chunk score: {score.toFixed(2)}</span>
      </div>
      {chunkIds.length > 0 && (
        <p className="mt-1 text-xs text-slate-400">Evidence chunks: {chunkIds.map((id) => `#${id}`).join(', ')}</p>
      )}
    </li>
  )
}

export function RelationshipList({ items }: { items: RelationshipVerification[] }) {
  if (items.length === 0) return <p className="text-sm text-slate-400">No relationships to verify.</p>
  return (
    <ul className="divide-y divide-slate-100">
      {items.map((r) => (
        <VerificationRow
          key={r.relationship_index}
          label={
            <>
              <strong>{r.subject}</strong> <span className="text-indigo-600">→ {r.relation} →</span>{' '}
              <strong>{r.object}</strong>
            </>
          }
          score={r.score}
          chunkIds={r.evidence_chunk_ids}
        />
      ))}
    </ul>
  )
}

export function RequirementList({ items }: { items: RequirementVerification[] }) {
  if (items.length === 0) return <p className="text-sm text-slate-400">No requirements to verify.</p>
  return (
    <ul className="divide-y divide-slate-100">
      {items.map((r) => (
        <VerificationRow key={r.requirement_index} label={r.requirement} score={r.score} chunkIds={r.evidence_chunk_ids} />
      ))}
    </ul>
  )
}

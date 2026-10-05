import type { ReactNode } from 'react'
import { Badge } from '@/components/ui'
import { verificationTone } from '@/lib/tone'
import type { RelationshipVerification, RequirementVerification } from '@/schemas/pipeline'

interface VerificationRowProps {
  label: ReactNode
  status: string
  score: number
  explanation?: string | null
  chunkIds: number[]
}

function VerificationRow({ label, status, score, explanation, chunkIds }: VerificationRowProps) {
  return (
    <li className="py-2.5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1 text-sm text-slate-800">{label}</div>
        <div className="flex shrink-0 items-center gap-1.5">
          <Badge tone={verificationTone(status)}>{status.replaceAll('_', ' ')}</Badge>
          <span className="text-xs text-slate-500 tabular-nums">{score.toFixed(2)}</span>
        </div>
      </div>
      {explanation && <p className="mt-1 text-sm text-slate-500">{explanation}</p>}
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
          status={r.status}
          score={r.score}
          explanation={r.explanation}
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
        <VerificationRow
          key={r.requirement_index}
          label={r.requirement}
          status={r.supported ? 'SUPPORTED' : 'NOT_SUPPORTED'}
          score={r.score}
          explanation={r.explanation}
          chunkIds={r.evidence_chunk_ids}
        />
      ))}
    </ul>
  )
}

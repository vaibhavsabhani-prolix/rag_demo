import { PatentHeading, ScoreBadge } from '@/components/patent'
import { Badge, Card } from '@/components/ui'
import { formatPercent } from '@/lib/format'
import type { FinalPatent } from '@/schemas/pipeline'

interface ResultCardProps {
  patent: FinalPatent
  rank: number
  onOpen: (patentId: string) => void
}

export function ResultCard({ patent, rank, onOpen }: ResultCardProps) {
  const topChunk = patent.evidence[0]

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onOpen(patent.patent_id)}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onOpen(patent.patent_id))}
      className="cursor-pointer p-5 transition hover:border-indigo-300 hover:shadow-md focus-visible:outline-2 focus-visible:outline-indigo-500"
    >
      <div className="flex items-start justify-between gap-4">
        <PatentHeading patentId={patent.patent_id} metadata={patent.metadata} rank={rank} />
        <ScoreBadge score={patent.final_score} />
      </div>

      {topChunk?.text && (
        <p className="mt-3 line-clamp-2 text-sm leading-relaxed text-slate-600">{topChunk.text}</p>
      )}

      <div className="mt-4 flex flex-wrap gap-1.5">
        <Badge tone="success">Relationships {formatPercent(patent.relationship_coverage)}</Badge>
        <Badge tone="info">Requirements {formatPercent(patent.requirement_coverage)}</Badge>
        <Badge tone="brand">Rerank {patent.best_reranker_score.toFixed(3)}</Badge>
        <Badge>Vector {patent.candidate_score.toFixed(3)}</Badge>
        {patent.contradicted_count > 0 && <Badge tone="danger">{patent.contradicted_count} contradicted</Badge>}
      </div>
    </Card>
  )
}

import { useCallback } from 'react'
import { EvidenceChunkCard, PatentHeading, RelationshipList, RequirementList, ScoreBadge } from '@/components/patent'
import { Drawer, JsonView, KeyValueList, ScoreBar, Section } from '@/components/ui'
import { formatValue } from '@/lib/format'
import { useAppDispatch, useAppSelector } from '@/store'
import type { SearchRun } from '@/store/searchSlice'
import { patentSelected } from '@/store/uiSlice'

const MAX_EVIDENCE = 8
const NO_WEIGHTS: Record<string, number> = {}

export function PatentDetailDrawer({ run }: { run: SearchRun }) {
  const dispatch = useAppDispatch()
  const selectedId = useAppSelector((s) => s.ui.selectedPatentId)
  const final = run.phases[7]?.data
  const patent = final?.results.find((p) => p.patent_id === selectedId)
  const weights = final?.weights_used ?? NO_WEIGHTS
  const onClose = useCallback(() => dispatch(patentSelected(null)), [dispatch])

  if (!patent) return null
  const sb = patent.score_breakdown
  // Each component's weighted value is out of (weight * 10) on the 0–10 scale.
  const maxFor = (key: string) => +((weights[key] ?? 0) * 10).toFixed(2)

  return (
    <Drawer open onClose={onClose} title={patent.patent_id} subtitle="Score breakdown, verification and evidence">
      <div className="space-y-8">
        <div className="flex items-start justify-between gap-4">
          <PatentHeading patentId={patent.patent_id} metadata={patent.metadata} />
          <ScoreBadge score={patent.final_score} />
        </div>

        <Section title="Score breakdown">
          <div className="space-y-3">
            <ScoreBar label="Relationships" value={sb.relationship_weighted} max={maxFor('relationship')} tone="success" />
            <ScoreBar label="Requirements" value={sb.requirement_weighted} max={maxFor('requirement')} tone="info" />
            <ScoreBar label="Cross-encoder rerank" value={sb.reranker_weighted} max={maxFor('reranker')} tone="brand" />
            <ScoreBar label="Vector retrieval" value={sb.retrieval_weighted} max={maxFor('retrieval')} tone="warning" />
          </div>
        </Section>

        <Section title={`Relationships (${patent.supported_count} supported)`}>
          <RelationshipList items={patent.relationships} />
        </Section>

        <Section title="Requirements">
          <RequirementList items={patent.requirements} />
        </Section>

        <Section title={`Top evidence (${Math.min(MAX_EVIDENCE, patent.evidence.length)} of ${patent.evidence.length})`}>
          <div className="space-y-2">
            {patent.evidence.slice(0, MAX_EVIDENCE).map((chunk) => (
              <EvidenceChunkCard
                key={chunk.chunk_id}
                chunkId={chunk.chunk_id}
                text={chunk.text}
                section={chunk.section}
                source={chunk.retrieval_source}
                rerankerScore={chunk.reranker_score}
                highlight={chunk.highlight}
              />
            ))}
          </div>
        </Section>

        <Section title="Metadata">
          <KeyValueList
            items={Object.entries(patent.metadata).map(([label, value]) => ({ label, value: formatValue(value) }))}
          />
        </Section>

        <JsonView data={patent} />
      </div>
    </Drawer>
  )
}

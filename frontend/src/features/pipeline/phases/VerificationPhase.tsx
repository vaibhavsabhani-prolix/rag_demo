import { RelationshipList, RequirementList } from '@/components/patent'
import { Badge, Collapsible, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs, formatPercent } from '@/lib/format'
import { unitScoreTone } from '@/lib/tone'
import type { PhaseResults } from '@/schemas/pipeline'

export function VerificationPhase({ data }: { data: PhaseResults[4] }) {
  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Evaluated" value={data.total_evaluated} hint={formatMs(data.timings.total_ms)} />
        <Stat label="Fully supported" value={data.fully_supported_count} />
        <Stat label="Partially supported" value={data.partially_supported_count} />
        <Stat label="Unsupported" value={data.unsupported_count} />
      </StatGrid>

      <Section title="Per-patent verification">
        <div className="space-y-2">
          {data.verified_patents.map((vp) => (
            <Collapsible
              key={vp.patent_id}
              title={<span className="font-mono">{vp.patent_id}</span>}
              aside={
                <div className="flex gap-1.5">
                  <Badge tone={unitScoreTone(vp.relationship_coverage)}>
                    Rel {formatPercent(vp.relationship_coverage)}
                  </Badge>
                  <Badge tone={unitScoreTone(vp.requirement_coverage)}>
                    Req {formatPercent(vp.requirement_coverage)}
                  </Badge>
                </div>
              }
            >
              <div className="space-y-5">
                <Section title="Relationships">
                  <RelationshipList items={vp.relationships} />
                </Section>
                <Section title="Requirements">
                  <RequirementList items={vp.requirements} />
                </Section>
              </div>
            </Collapsible>
          ))}
        </div>
      </Section>
    </div>
  )
}

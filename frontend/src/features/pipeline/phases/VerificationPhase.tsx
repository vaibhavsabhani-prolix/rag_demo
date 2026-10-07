import { RelationshipList, RequirementList } from '@/components/patent'
import { Collapsible, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs } from '@/lib/format'
import type { PatentVerification, PhaseResults } from '@/schemas/pipeline'

function VerifiedPatentCard({ vp }: { vp: PatentVerification }) {
  return (
    <Collapsible
      key={vp.patent_id}
      title={<span className="font-mono">{vp.patent_id}</span>}
      aside={
        <div className="flex items-center gap-1.5 text-xs font-medium tabular-nums">
          <span className="rounded-full bg-indigo-50 px-2 py-1 text-indigo-700">
            Relationship: {vp.relationship_coverage.toFixed(2)}
          </span>
          <span className="rounded-full bg-indigo-50 px-2 py-1 text-indigo-700">
            Requirement: {vp.requirement_coverage.toFixed(2)}
          </span>
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
  )
}

export function VerificationPhase({ data }: { data: PhaseResults[4] }) {
  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Evaluated" value={data.total_evaluated} hint={formatMs(data.timings.total_ms)} />
        <Stat label="Fully supported" value={data.fully_supported_count} />
        <Stat label="Partially supported" value={data.partially_supported_count} />
        <Stat label="Unsupported" value={data.unsupported_count} />
      </StatGrid>

      <Section title={`Evaluated patents (${data.verified_patents.length})`}>
        <div className="space-y-2">
          {data.verified_patents.map((vp) => (
            <VerifiedPatentCard key={vp.patent_id} vp={vp} />
          ))}
        </div>
      </Section>
    </div>
  )
}

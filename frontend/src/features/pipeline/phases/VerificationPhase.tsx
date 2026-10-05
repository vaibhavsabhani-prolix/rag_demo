import { RelationshipList, RequirementList } from '@/components/patent'
import { Badge, Collapsible, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs } from '@/lib/format'
import type { PatentVerification, PhaseResults } from '@/schemas/pipeline'

function VerifiedPatentCard({ vp, qualified }: { vp: PatentVerification; qualified: boolean }) {
  return (
    <Collapsible
      key={vp.patent_id}
      title={<span className="font-mono">{vp.patent_id}</span>}
      aside={<Badge tone={qualified ? 'success' : 'danger'}>{qualified ? 'QUALIFIED' : 'REJECTED'}</Badge>}
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

      <Section title={`Qualified patents (${data.verified_patents.length})`}>
        <div className="space-y-2">
          {data.verified_patents.map((vp) => (
            <VerifiedPatentCard key={vp.patent_id} vp={vp} qualified />
          ))}
        </div>
      </Section>

      {data.eliminated_patents.length > 0 && (
        <Section title={`Rejected patents (${data.eliminated_patents.length})`}>
          <div className="space-y-2">
            {data.eliminated_patents.map((vp) => (
              <VerifiedPatentCard key={vp.patent_id} vp={vp} qualified={false} />
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}

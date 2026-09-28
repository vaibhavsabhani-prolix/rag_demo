import { EvidenceChunkCard } from '@/components/patent'
import { Badge, Collapsible, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs } from '@/lib/format'
import { summarizePatent } from '@/lib/patent'
import type { PhaseResults } from '@/schemas/pipeline'

export function EvidencePhase({ data }: { data: PhaseResults[4] }) {
  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Candidates" value={data.total_candidates} />
        <Stat label="Evidence chunks" value={data.total_evidence_chunks} />
        <Stat label="Qdrant" value={formatMs(data.timings.qdrant_retrieval_ms)} />
        <Stat label="Total time" value={formatMs(data.timings.total_ms)} />
      </StatGrid>

      <Section title="Evidence query">
        <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">{data.evidence_query_text || '—'}</p>
      </Section>

      <Section title="Evidence by patent">
        <div className="space-y-2">
          {data.patent_evidence_list.map((pe) => (
            <Collapsible
              key={pe.patent_id}
              title={<span className="font-mono">{pe.patent_id}</span>}
              subtitle={summarizePatent(pe.metadata).title}
              aside={<Badge>{pe.chunks.length} chunks</Badge>}
            >
              <div className="space-y-2">
                {pe.chunks.map((c) => (
                  <EvidenceChunkCard
                    key={c.chunk_id}
                    chunkId={c.chunk_id}
                    text={c.text}
                    section={c.section}
                    source={c.retrieval_source}
                    retrievalScore={c.retrieval_score}
                  />
                ))}
              </div>
            </Collapsible>
          ))}
        </div>
      </Section>
    </div>
  )
}

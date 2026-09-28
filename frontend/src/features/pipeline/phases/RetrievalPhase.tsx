import { KeyValueList, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs } from '@/lib/format'
import type { PhaseResults } from '@/schemas/pipeline'
import { CandidateTable } from './CandidateTable'

export function RetrievalPhase({ data }: { data: PhaseResults[2] }) {
  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Views" value={Object.keys(data.retrieval_views).length} />
        <Stat label="Chunk hits" value={data.total_chunk_hits} />
        <Stat label="Unique patents" value={data.unique_patents} />
        <Stat
          label="Total time"
          value={formatMs(data.timings.total_ms)}
          hint={`embed ${formatMs(data.timings.embedding_ms)} · qdrant ${formatMs(data.timings.qdrant_retrieval_ms)}`}
        />
      </StatGrid>

      <Section title="Retrieval views">
        <KeyValueList items={Object.entries(data.retrieval_views).map(([label, value]) => ({ label, value }))} />
      </Section>

      <Section title="Candidates">
        <CandidateTable candidates={data.candidates} />
      </Section>
    </div>
  )
}

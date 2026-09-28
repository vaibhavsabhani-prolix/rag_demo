import { Badge, DataTable, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs } from '@/lib/format'
import { unitScoreTone } from '@/lib/tone'
import type { PhaseResults } from '@/schemas/pipeline'

export function RerankPhase({ data }: { data: PhaseResults[6] }) {
  const patents = [...data.reranked_patents].sort((a, b) => b.best_reranker_score - a.best_reranker_score)

  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Candidates" value={data.total_candidates} />
        <Stat label="Chunks scored" value={data.total_chunks_reranked} hint={`${data.truncated_chunks_count} truncated`} />
        <Stat label="Requests" value={data.total_requests} />
        <Stat label="Total time" value={formatMs(data.timings.total_ms)} />
      </StatGrid>

      <Section title="Reranking query">
        <p className="rounded-lg bg-slate-50 p-3 font-mono text-xs text-slate-700">{data.reranking_query || '—'}</p>
      </Section>

      <Section title="Patents by best rerank score">
        <DataTable
          rows={patents}
          rowKey={(p) => p.patent_id}
          emptyText="No patents reranked."
          columns={[
            { header: 'Patent', render: (p) => <span className="font-mono">{p.patent_id}</span> },
            {
              header: 'Best',
              render: (p) => <Badge tone={unitScoreTone(p.best_reranker_score)}>{p.best_reranker_score.toFixed(4)}</Badge>,
              align: 'right',
            },
            { header: 'Average', render: (p) => p.avg_reranker_score.toFixed(4), align: 'right' },
            { header: 'Chunks', render: (p) => p.evidence.length, align: 'right' },
            { header: 'Supported rel.', render: (p) => p.supported_count, align: 'right' },
          ]}
        />
      </Section>
    </div>
  )
}

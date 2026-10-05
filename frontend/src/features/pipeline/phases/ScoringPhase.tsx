import { DataTable, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs, formatPercent } from '@/lib/format'
import type { PhaseResults } from '@/schemas/pipeline'

export function ScoringPhase({ data }: { data: PhaseResults[6] }) {
  // A metadata-only query has nothing semantic to score or rank by (see app/scoring/scorer.py).
  const isMetadataOnly = data.results.length > 0 && data.results.every((r) => r.final_score === null)

  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Evaluated" value={data.total_candidates_evaluated} />
        <Stat
          label={isMetadataOnly ? 'Matched' : `Passed (≥ ${data.threshold_used})`}
          value={data.passed_threshold_count}
        />
        <Stat label="Excluded" value={data.rejected_count} />
        <Stat label="Time" value={formatMs(data.timings.total_ms)} />
      </StatGrid>

      <Section
        title={
          isMetadataOnly
            ? 'Not scored: this query is a metadata filter only, with nothing semantic to rank by.'
            : `Weights: ${Object.entries(data.weights_used)
                .map(([k, w]) => `${k} ${formatPercent(w)}`)
                .join(' · ')}`
        }
      >
        <DataTable
          rows={data.results}
          rowKey={(r) => r.patent_id}
          emptyText="No patent passed the threshold."
          columns={[
            { header: '#', render: (_, i) => i + 1, align: 'right' },
            { header: 'Patent', render: (r) => <span className="font-mono">{r.patent_id}</span> },
            {
              header: 'Final',
              render: (r) => <strong>{r.final_score === null ? '—' : r.final_score.toFixed(2)}</strong>,
              align: 'right',
            },
            {
              header: 'Rel.',
              render: (r) => r.score_breakdown?.relationship_weighted.toFixed(2) ?? '—',
              align: 'right',
            },
            {
              header: 'Req.',
              render: (r) => r.score_breakdown?.requirement_weighted.toFixed(2) ?? '—',
              align: 'right',
            },
            {
              header: 'Rerank',
              render: (r) => r.score_breakdown?.reranker_weighted.toFixed(2) ?? '—',
              align: 'right',
            },
            {
              header: 'Vector',
              render: (r) => r.score_breakdown?.retrieval_weighted.toFixed(2) ?? '—',
              align: 'right',
            },
          ]}
        />
      </Section>
    </div>
  )
}

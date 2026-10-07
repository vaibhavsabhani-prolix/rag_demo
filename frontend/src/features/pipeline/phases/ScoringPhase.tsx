import { DataTable, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs, formatPercent } from '@/lib/format'
import type { PhaseResults } from '@/schemas/pipeline'

/** One weighted score column: the weighted contribution, with "weight × raw score" underneath so the math is visible without opening raw JSON. */
function WeightedCell({ weighted, raw, weight }: { weighted: number; raw: number; weight: number }) {
  return (
    <div title={`${formatPercent(weight)} × ${raw.toFixed(2)} × 10 = ${weighted.toFixed(2)}`}>
      <div className="tabular-nums">{weighted.toFixed(2)}</div>
      <div className="text-[11px] text-slate-400 tabular-nums">
        {formatPercent(weight)} × {raw.toFixed(2)}
      </div>
    </div>
  )
}

export function ScoringPhase({ data }: { data: PhaseResults[6] }) {
  // A metadata-only query has nothing semantic to score or rank by (see app/scoring/scorer.py).
  const isMetadataOnly = data.results.length > 0 && data.results.every((r) => r.final_score === null)
  const weights = data.weights_used

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
            : `Weights: ${Object.entries(weights)
                .map(([k, w]) => `${k} ${formatPercent(w)}`)
                .join(' · ')}`
        }
      >
        {!isMetadataOnly && (
          <p className="text-xs text-slate-400">
            Each column = weight × raw score (0–1) × 10, hover a cell for the exact equation. Final = Rel. + Req. + Rerank + Vector.
          </p>
        )}
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
              render: (r) =>
                r.score_breakdown ? (
                  <WeightedCell
                    weighted={r.score_breakdown.relationship_weighted}
                    raw={r.score_breakdown.relationship_score}
                    weight={weights.relationship ?? 0}
                  />
                ) : (
                  '—'
                ),
              align: 'right',
            },
            {
              header: 'Req.',
              render: (r) =>
                r.score_breakdown ? (
                  <WeightedCell
                    weighted={r.score_breakdown.requirement_weighted}
                    raw={r.score_breakdown.requirement_score}
                    weight={weights.requirement ?? 0}
                  />
                ) : (
                  '—'
                ),
              align: 'right',
            },
            {
              header: 'Rerank',
              render: (r) =>
                r.score_breakdown ? (
                  <WeightedCell
                    weighted={r.score_breakdown.reranker_weighted}
                    raw={r.score_breakdown.reranker_score}
                    weight={weights.reranker ?? 0}
                  />
                ) : (
                  '—'
                ),
              align: 'right',
            },
            {
              header: 'Vector',
              render: (r) =>
                r.score_breakdown ? (
                  <WeightedCell
                    weighted={r.score_breakdown.retrieval_weighted}
                    raw={r.score_breakdown.retrieval_score}
                    weight={weights.retrieval ?? 0}
                  />
                ) : (
                  '—'
                ),
              align: 'right',
            },
          ]}
        />
      </Section>
    </div>
  )
}

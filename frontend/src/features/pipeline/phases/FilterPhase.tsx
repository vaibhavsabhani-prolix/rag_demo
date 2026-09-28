import { DataTable, Section, Stat, StatGrid } from '@/components/ui'
import { formatMs, formatValue } from '@/lib/format'
import type { PhaseResults } from '@/schemas/pipeline'
import { CandidateTable } from './CandidateTable'

export function FilterPhase({ data }: { data: PhaseResults[3] }) {
  return (
    <div className="space-y-6">
      <StatGrid>
        <Stat label="Before" value={data.total_before} />
        <Stat label="After" value={data.total_after} />
        <Stat label="Rejected" value={data.filtered_count} />
        <Stat label="Time" value={formatMs(data.filter_time_ms)} />
      </StatGrid>

      <Section title="Filter diagnostics">
        <DataTable
          rows={data.diagnostics}
          rowKey={(_, i) => String(i)}
          emptyText="No metadata filters were applied."
          columns={[
            { header: 'Field', render: (d) => <code className="font-mono">{d.field}</code> },
            { header: 'Operator', render: (d) => d.operator },
            { header: 'Value', render: (d) => formatValue(d.value) },
            { header: 'Passed', render: (d) => d.passed, align: 'right' },
            { header: 'Failed', render: (d) => d.failed, align: 'right' },
          ]}
        />
      </Section>

      <Section title="Surviving candidates">
        <CandidateTable candidates={data.candidates} />
      </Section>
    </div>
  )
}

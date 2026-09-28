import { Badge, BulletList, DataTable, Section, TagList } from '@/components/ui'
import { formatValue } from '@/lib/format'
import type { ParsedQuery } from '@/schemas/pipeline'

export function QueryPhase({ data, cacheHit }: { data: ParsedQuery; cacheHit: boolean }) {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-1.5">
        <Badge tone={cacheHit ? 'success' : 'info'}>{cacheHit ? 'Cache hit' : 'LLM call'}</Badge>
        {data.is_metadata_only && <Badge tone="warning">Metadata-only query</Badge>}
      </div>

      <Section title="Semantic query">
        <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">{data.semantic_query || '—'}</p>
      </Section>

      <Section title="Concepts">
        <TagList items={data.concepts} tone="brand" />
      </Section>

      <Section title="Relationships">
        <DataTable
          rows={data.relationships}
          rowKey={(_, i) => String(i)}
          emptyText="No relationships extracted."
          columns={[
            { header: 'Subject', render: (r) => r.subject },
            { header: 'Relation', render: (r) => <span className="text-indigo-600">{r.relation}</span> },
            { header: 'Object', render: (r) => r.object },
            { header: 'Context', render: (r) => r.context ?? '—' },
          ]}
        />
      </Section>

      {data.attributes.length > 0 && (
        <Section title="Attributes">
          <DataTable
            rows={data.attributes}
            rowKey={(_, i) => String(i)}
            columns={[
              { header: 'Concept', render: (a) => a.concept },
              { header: 'Attribute', render: (a) => a.name },
              { header: 'Value', render: (a) => a.value },
            ]}
          />
        </Section>
      )}

      <div className="grid gap-6 md:grid-cols-3">
        <Section title="Requirements">
          <BulletList items={data.requirements} />
        </Section>
        <Section title="Constraints">
          <BulletList items={data.constraints} />
        </Section>
        <Section title="Exclusions">
          <BulletList items={data.exclusions} />
        </Section>
      </div>

      <Section title="Metadata filters">
        <DataTable
          rows={data.metadata_filters}
          rowKey={(_, i) => String(i)}
          emptyText="No metadata filters."
          columns={[
            { header: 'Field', render: (f) => <code className="font-mono">{f.field}</code> },
            { header: 'Operator', render: (f) => f.operator },
            { header: 'Value', render: (f) => formatValue(f.value) },
            { header: 'From', render: (f) => f.raw_field ?? '—' },
          ]}
        />
      </Section>
    </div>
  )
}

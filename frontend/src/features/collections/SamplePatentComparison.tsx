import { Card, CardBody, CardHeader, type Column, DataTable } from '@/components/ui'
import { formatBytes, formatCompact } from '@/lib/format'
import { summarizePatent } from '@/lib/patent'
import type { SamplePatentStats, SamplePatentCollectionChunks } from '@/schemas/api'

/**
 * The same patent indexed into every chunk-size collection, with its chunk
 * count and estimated storage footprint shown side by side so the effect of
 * MAX_CHUNK_TOKENS on one real patent is visible directly.
 */
export function SamplePatentComparison({ patent }: { patent: SamplePatentStats }) {
  const { title, assignee, year, country } = summarizePatent(patent.metadata)

  const columns: Column<SamplePatentCollectionChunks>[] = [
    {
      header: 'Collection',
      render: (row) => <span className="font-medium text-slate-900">{row.collection}</span>,
    },
    { header: 'Chunks', align: 'right', render: (row) => formatCompact(row.chunk_count) },
    { header: 'Vector bytes', align: 'right', render: (row) => formatBytes(row.vector_bytes) },
    { header: 'Payload (text) bytes', align: 'right', render: (row) => formatBytes(row.payload_bytes) },
    {
      header: 'Estimated total',
      align: 'right',
      render: (row) => <span className="font-semibold text-slate-900">{formatBytes(row.estimated_total_bytes)}</span>,
    },
  ]

  return (
    <Card>
      <CardHeader
        title={title || patent.patent_id}
        subtitle={
          <span>
            {patent.patent_id} · {assignee}
            {year ? ` · ${year}` : ''}
            {country ? ` · ${country}` : ''}
          </span>
        }
      />
      <CardBody className="space-y-3">
        <DataTable
          columns={columns}
          rows={patent.per_collection}
          rowKey={(row) => row.collection}
          emptyText="Not found in any collection."
        />
        <p className="text-xs text-slate-500">
          Same underlying patent text, split differently: a smaller chunk-token limit produces more, smaller chunks
          for the same patent, so more vectors (the main driver of the storage difference) are stored for it.
          "Estimated total" is vector bytes (float32 × vector dimension × chunks) plus the actual chunk payload size
          — a raw-data estimate, not Qdrant's on-disk size, which also includes HNSW index, WAL and segment overhead.
        </p>
      </CardBody>
    </Card>
  )
}

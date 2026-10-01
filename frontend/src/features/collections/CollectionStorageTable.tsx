import { Card, CardBody, CardHeader, type Column, DataTable } from '@/components/ui'
import { formatBytes, formatCompact } from '@/lib/format'
import type { CollectionStorageStats } from '@/schemas/api'

/** One row per chunk-size collection: how much it holds, and what Qdrant reports for it. */
export function CollectionStorageTable({ collections }: { collections: CollectionStorageStats[] }) {
  const columns: Column<CollectionStorageStats>[] = [
    {
      header: 'Collection',
      render: (c) => (
        <div>
          <div className="font-medium text-slate-900">{c.name}</div>
          <div className="text-xs text-slate-500">
            {c.max_chunk_tokens ? `max ${formatCompact(c.max_chunk_tokens)} tokens/chunk` : c.chunks_collection}
          </div>
        </div>
      ),
    },
    {
      header: 'Patents',
      align: 'right',
      render: (c) => formatCompact(c.patent_count),
    },
    {
      header: 'Chunks',
      align: 'right',
      render: (c) => formatCompact(c.chunk_count),
    },
    {
      header: 'Avg chunks/patent',
      align: 'right',
      render: (c) => (c.patent_count ? (c.chunk_count / c.patent_count).toFixed(1) : '—'),
    },
    {
      header: 'Disk usage',
      align: 'right',
      render: (c) => (
        <div>
          <div className="font-medium text-slate-900">{formatBytes(c.total_disk_bytes)}</div>
          <div className="text-xs text-slate-500">
            chunks {formatBytes(c.chunks_memory.disk_bytes)} · patents {formatBytes(c.patents_memory.disk_bytes)}
          </div>
        </div>
      ),
    },
    {
      header: 'RAM usage',
      align: 'right',
      render: (c) => (
        <div>
          <div className="font-medium text-slate-900">{formatBytes(c.total_ram_bytes)}</div>
          <div className="text-xs text-slate-500">
            chunks {formatBytes(c.chunks_memory.ram_bytes)} · patents {formatBytes(c.patents_memory.ram_bytes)}
          </div>
        </div>
      ),
    },
  ]

  return (
    <Card>
      <CardHeader
        title="Collections"
        subtitle="One collection pair per chunk-size variant. Disk/RAM figures are Qdrant's own reported usage for each collection, not an estimate."
      />
      <CardBody>
        <DataTable columns={columns} rows={collections} rowKey={(c) => c.name} emptyText="No collections indexed yet." />
      </CardBody>
    </Card>
  )
}

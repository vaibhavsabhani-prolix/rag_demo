import { DataTable, TagList } from '@/components/ui'
import { summarizePatent } from '@/lib/patent'
import type { CandidatePatent } from '@/schemas/pipeline'

/** Candidate patents table shared by the retrieval and filtering phases. */
export function CandidateTable({ candidates }: { candidates: CandidatePatent[] }) {
  return (
    <DataTable
      rows={candidates}
      rowKey={(c) => c.patent_id}
      emptyText="No candidates."
      columns={[
        { header: '#', render: (_, i) => i + 1, align: 'right' },
        { header: 'Patent', render: (c) => <span className="font-mono">{c.patent_id}</span> },
        {
          header: 'Title',
          render: (c) => <span className="line-clamp-1">{summarizePatent(c.metadata).title || '—'}</span>,
          className: 'max-w-xs',
        },
        { header: 'Score', render: (c) => c.retrieval_score.toFixed(4), align: 'right' },
        { header: 'Chunks', render: (c) => c.chunk_count, align: 'right' },
        { header: 'Views', render: (c) => <TagList items={c.matched_views} /> },
      ]}
    />
  )
}

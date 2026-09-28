import { useState } from 'react'
import { Badge } from '@/components/ui'
import { formatScore } from '@/lib/format'
import { unitScoreTone } from '@/lib/tone'

interface EvidenceChunkCardProps {
  chunkId: number
  text?: string | null
  section?: string | null
  source?: string
  retrievalScore?: number
  rerankerScore?: number
}

const PREVIEW_CHARS = 360

/** One patent text chunk with its scores. Collapsed, it shows the start of the text; expanded, the full chunk. */
export function EvidenceChunkCard({ chunkId, text, section, source, retrievalScore, rerankerScore }: EvidenceChunkCardProps) {
  const [expanded, setExpanded] = useState(false)
  const body = text ?? ''
  const isLong = body.length > PREVIEW_CHARS

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="text-sm font-medium text-slate-800">Chunk #{chunkId}</span>
        {section && <Badge>{section}</Badge>}
        {source && <Badge tone="info">{source.replaceAll('_', ' ')}</Badge>}
        <span className="flex-1" />
        {retrievalScore !== undefined && <Badge tone="neutral">vector {formatScore(retrievalScore)}</Badge>}
        {rerankerScore !== undefined && (
          <Badge tone={unitScoreTone(rerankerScore)}>rerank {formatScore(rerankerScore)}</Badge>
        )}
      </div>
      {body ? (
        <p className="text-sm leading-relaxed whitespace-pre-line text-slate-700">
          {expanded || !isLong ? body : `${body.slice(0, PREVIEW_CHARS)}…`}
          {isLong && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="ml-1 font-medium text-indigo-600 hover:underline"
            >
              {expanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </p>
      ) : (
        <p className="text-sm text-slate-400 italic">No text loaded.</p>
      )}
    </div>
  )
}

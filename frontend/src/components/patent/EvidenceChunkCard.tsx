import clsx from 'clsx'
import { useState, type ReactNode } from 'react'
import { Badge } from '@/components/ui'
import { formatScore } from '@/lib/format'
import { unitScoreTone } from '@/lib/tone'
import type { ChunkHighlight } from '@/schemas/pipeline'

interface EvidenceChunkCardProps {
  chunkId: number
  text?: string | null
  section?: string | null
  source?: string
  retrievalScore?: number
  rerankerScore?: number
  /** Sentences and words of the text that match the query (from Phase 6 reranking). */
  highlight?: ChunkHighlight | null
}

const PREVIEW_CHARS = 360

/** One patent text chunk with its scores. Collapsed, it shows the start of the text; expanded, the full chunk. */
export function EvidenceChunkCard({
  chunkId,
  text,
  section,
  source,
  retrievalScore,
  rerankerScore,
  highlight,
}: EvidenceChunkCardProps) {
  const [expanded, setExpanded] = useState(false)
  const body = text ?? ''
  const isLong = body.length > PREVIEW_CHARS
  const matchCount = highlight?.sentences.length ?? 0

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="text-sm font-medium text-slate-800">Chunk #{chunkId}</span>
        {section && <Badge>{section}</Badge>}
        {source && <Badge tone="info">{source.replaceAll('_', ' ')}</Badge>}
        {highlight && (
          <Badge tone={matchCount ? 'warning' : 'neutral'}>
            {matchCount} matching sentence{matchCount === 1 ? '' : 's'}
          </Badge>
        )}
        <span className="flex-1" />
        {retrievalScore !== undefined && <Badge tone="neutral">vector {formatScore(retrievalScore)}</Badge>}
        {rerankerScore !== undefined && (
          <Badge tone={unitScoreTone(rerankerScore)}>rerank {formatScore(rerankerScore)}</Badge>
        )}
      </div>
      {body ? (
        <p className="text-sm leading-relaxed whitespace-pre-line text-slate-700">
          <HighlightedText text={body} limit={expanded || !isLong ? body.length : PREVIEW_CHARS} highlight={highlight} />
          {isLong && !expanded && '…'}
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

type Sentence = ChunkHighlight['sentences'][number]

/** `text` up to `limit` characters, with matching sentences marked and query words underlined. */
function HighlightedText({ text, limit, highlight }: { text: string; limit: number; highlight?: ChunkHighlight | null }) {
  if (!highlight) return text.slice(0, limit)

  // Cut the text at every sentence and term edge, then render each piece.
  const cuts = new Set([0, limit])
  for (const s of [...highlight.sentences, ...highlight.terms]) {
    if (s.start < limit) cuts.add(s.start)
    if (s.end < limit) cuts.add(s.end)
  }
  const edges = [...cuts].sort((a, b) => a - b)

  // Consecutive pieces inside the same sentence are grouped under one <mark>.
  const groups: { sentence?: Sentence; pieces: ReactNode[] }[] = []
  for (let i = 0; i < edges.length - 1; i++) {
    const [a, b] = [edges[i], edges[i + 1]]
    const sentence = highlight.sentences.find((s) => s.start <= a && a < s.end)
    const isTerm = highlight.terms.some((t) => t.start <= a && a < t.end)
    const piece = isTerm ? (
      <span key={a} className="hl-term">
        {text.slice(a, b)}
      </span>
    ) : (
      text.slice(a, b)
    )
    const last = groups.at(-1)
    if (last && last.sentence === sentence) last.pieces.push(piece)
    else groups.push({ sentence, pieces: [piece] })
  }

  return groups.map(({ sentence, pieces }, i) =>
    sentence ? (
      <mark
        key={i}
        title={`Matches the query in meaning (rerank ${formatScore(sentence.score)})`}
        className={clsx('hl-sentence', sentence.strong && 'hl-sentence-strong')}
      >
        {pieces}
      </mark>
    ) : (
      <span key={i}>{pieces}</span>
    ),
  )
}

import { useState } from 'react'
import { Badge, ChevronIcon } from '@/components/ui'
import type { FeatureSearchResult, FeatureChunkResult } from '@/schemas/keyfeature'

interface KeyFeatureCardProps {
  result: FeatureSearchResult
  defaultExpanded?: boolean
}

export function KeyFeatureCard({ result, defaultExpanded = true }: KeyFeatureCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const chunks = result.chunks || []

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-xs transition-all">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded((prev) => !prev)}
        className="flex w-full items-start justify-between gap-4 p-4 text-left transition-colors hover:bg-slate-50/80 sm:items-center sm:p-5"
      >
        <div className="flex items-start gap-3 sm:items-center">
          <Badge tone="brand" className="shrink-0 font-mono text-xs font-bold">
            Feature {result.feature_id}
          </Badge>
          <div>
            <h3 className="text-base font-semibold text-slate-900">{result.feature}</h3>
            <p className="mt-0.5 text-xs text-slate-500">
              Independent embedding &amp; vector search · {chunks.length} candidate {chunks.length === 1 ? 'chunk' : 'chunks'} retrieved
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <Badge tone={chunks.length > 0 ? 'success' : 'neutral'} className="text-xs">
            {chunks.length} hits
          </Badge>
          <div
            className={`flex size-7 items-center justify-center rounded-md bg-slate-100 text-slate-600 transition-transform ${
              isExpanded ? 'rotate-90' : ''
            }`}
          >
            <ChevronIcon className="size-4" />
          </div>
        </div>
      </button>

      {/* Expanded Chunks Content */}
      {isExpanded && (
        <div className="border-t border-slate-100 bg-slate-50/50 p-4 sm:p-5">
          {chunks.length === 0 ? (
            <p className="text-sm italic text-slate-500">No matching chunks retrieved for this feature.</p>
          ) : (
            <div className="space-y-3">
              {chunks.map((chunk, index) => (
                <ChunkItem key={`${chunk.patent_id}-${chunk.chunk_id}-${index}`} chunk={chunk} index={index + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ChunkItem({ chunk, index }: { chunk: FeatureChunkResult; index: number }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(chunk.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-surface p-4 shadow-2xs transition-shadow hover:shadow-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-bold text-slate-400">#{index}</span>
          <a
            href={`https://patents.google.com/patent/${chunk.patent_id}/en`}
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-sm font-semibold text-indigo-600 hover:underline"
            title="Open in Google Patents"
          >
            {chunk.patent_id}
          </a>
          {chunk.title && (
            <span className="max-w-md truncate text-xs font-medium text-slate-700" title={chunk.title}>
              — {chunk.title}
            </span>
          )}
          {chunk.section && (
            <Badge tone="neutral" className="text-2xs uppercase">
              {chunk.section}
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Sim:</span>
            <span className="font-mono text-xs font-bold text-slate-800">{chunk.score.toFixed(3)}</span>
            <div className="w-16 h-2 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full bg-indigo-500 rounded-full"
                style={{ width: `${Math.min(100, Math.max(0, chunk.score * 100))}%` }}
              />
            </div>
          </div>
          <button
            type="button"
            onClick={handleCopy}
            className="rounded px-2 py-0.5 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-900"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      <div className="mt-2.5 text-xs leading-relaxed text-slate-700 whitespace-pre-wrap font-sans bg-slate-50/60 p-3 rounded-md border border-slate-100">
        {chunk.text}
      </div>

      <div className="mt-2 flex items-center justify-between text-2xs text-slate-400">
        <span>Chunk ID: {chunk.chunk_id}</span>
        {chunk.token_count && <span>Tokens: {chunk.token_count}</span>}
      </div>
    </div>
  )
}

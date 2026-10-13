import { useState } from 'react'
import { HighlightedText } from '@/components/patent/EvidenceChunkCard'
import { Badge, ChevronIcon } from '@/components/ui'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type { FeatureSearchResult, FeatureChunkResult, FeaturePatentResult } from '@/schemas/keyfeature'

interface KeyFeatureCardProps {
  result: FeatureSearchResult
  defaultExpanded?: boolean
}

function getPatents(result: FeatureSearchResult): FeaturePatentResult[] {
  if (result.patents && result.patents.length > 0) {
    return result.patents
  }
  // Fallback: group result.chunks by patent_id
  const map = new Map<string, FeaturePatentResult>()
  for (const chunk of result.chunks || []) {
    let p = map.get(chunk.patent_id)
    if (!p) {
      p = {
        patent_id: chunk.patent_id,
        title: chunk.title ?? null,
        score: chunk.score,
        best_reranker_score: chunk.reranker_score ?? null,
        best_retrieval_score: chunk.retrieval_score ?? null,
        metadata: chunk.metadata || {},
        chunks: [],
      }
      map.set(chunk.patent_id, p)
    }
    p.chunks.push(chunk)
    p.score = Math.max(p.score, chunk.score)
    if (chunk.reranker_score != null) {
      p.best_reranker_score = Math.max(p.best_reranker_score ?? 0, chunk.reranker_score)
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => (b.best_reranker_score ?? b.score) - (a.best_reranker_score ?? a.score),
  )
}

export function KeyFeatureCard({ result, defaultExpanded = true }: KeyFeatureCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const patents = getPatents(result)
  const totalChunks = patents.reduce((sum, p) => sum + (p.chunks?.length || 0), 0)

  // Track expanded state for each patent (default: expanded)
  const [expandedPatents, setExpandedPatents] = useState<Record<string, boolean>>({})

  const isPatentExpanded = (patentId: string) => expandedPatents[patentId] ?? true

  const togglePatent = (patentId: string) => {
    setExpandedPatents((prev) => ({
      ...prev,
      [patentId]: !(prev[patentId] ?? true),
    }))
  }

  const allPatentsCollapsed = patents.length > 0 && patents.every((p) => !(expandedPatents[p.patent_id] ?? true))

  const toggleAllPatents = () => {
    const nextState = allPatentsCollapsed
    const nextMap: Record<string, boolean> = {}
    for (const p of patents) {
      nextMap[p.patent_id] = nextState
    }
    setExpandedPatents(nextMap)
  }

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-xs transition-all">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded((prev) => !prev)}
        className="flex w-full items-start justify-between gap-4 p-4 text-left transition-colors hover:bg-slate-100/50 sm:items-center sm:p-5"
      >
        <div className="flex items-start gap-3 sm:items-center">
          <Badge tone="brand" className="shrink-0 font-mono text-xs font-bold">
            Feature {result.feature_id}
          </Badge>
          <div>
            <h3 className="text-base font-semibold text-slate-900">{result.feature}</h3>
            <p className="mt-0.5 text-xs text-slate-500">
              BGE Cross-Encoder Reranked · {patents.length} final {patents.length === 1 ? 'patent' : 'patents'} selected ({totalChunks} matching {totalChunks === 1 ? 'chunk' : 'chunks'})
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <Badge tone={patents.length > 0 ? 'success' : 'neutral'} className="text-xs">
            {patents.length} {patents.length === 1 ? 'patent' : 'patents'}
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

      {/* Expanded Patents Content */}
      {isExpanded && (
        <div className="border-t border-slate-200 bg-slate-50/50 p-4 sm:p-5">
          {patents.length === 0 ? (
            <p className="text-sm italic text-slate-500">No matching patents retrieved for this feature.</p>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-500 px-0.5 pb-0.5">
                <span className="font-semibold uppercase tracking-wider text-2xs text-slate-600">
                  {patents.length} Unique Patents
                </span>
                <button
                  type="button"
                  onClick={toggleAllPatents}
                  className="font-medium text-indigo-700 hover:text-indigo-800 hover:underline dark:text-indigo-400 dark:hover:text-indigo-300 transition-colors"
                >
                  {allPatentsCollapsed ? 'Expand all chunks' : 'Collapse all chunks'}
                </button>
              </div>
              {patents.map((patent, index) => (
                <PatentItem
                  key={`${patent.patent_id}-${index}`}
                  patent={patent}
                  rank={index + 1}
                  isExpanded={isPatentExpanded(patent.patent_id)}
                  onToggle={() => togglePatent(patent.patent_id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface PatentItemProps {
  patent: FeaturePatentResult
  rank: number
  isExpanded?: boolean
  onToggle?: () => void
}

function PatentItem({ patent, rank, isExpanded: propIsExpanded, onToggle: propOnToggle }: PatentItemProps) {
  const { data: config } = usePipelineConfig()
  const url = patentUrl(config?.patent_view_url_template, patent.patent_id)
  const chunks = patent.chunks || []

  // Internal state fallback if uncontrolled
  const [internalExpanded, setInternalExpanded] = useState(true)
  const isExpanded = propIsExpanded !== undefined ? propIsExpanded : internalExpanded
  const handleToggle = propOnToggle || (() => setInternalExpanded((prev) => !prev))

  return (
    <div className="rounded-xl border border-slate-200 bg-surface shadow-xs transition-shadow hover:shadow-sm overflow-hidden">
      {/* Patent Header Dropdown Toggle */}
      <div
        role="button"
        tabIndex={0}
        onClick={handleToggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            handleToggle()
          }
        }}
        className={`flex flex-wrap items-center justify-between gap-3 p-3.5 sm:px-4 cursor-pointer select-none transition-colors hover:bg-slate-100/60 ${
          isExpanded ? 'border-b border-slate-200/80 bg-slate-100/50' : 'bg-surface'
        }`}
      >
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="flex size-6 items-center justify-center rounded-md bg-indigo-50 font-mono text-xs font-bold text-indigo-700">
            #{rank}
          </span>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="font-mono text-sm font-bold text-indigo-700 hover:text-indigo-800 hover:underline dark:text-indigo-300 dark:hover:text-indigo-200"
            title="Open in Patent View"
          >
            {patent.patent_id}
          </a>
          {patent.title && (
            <span className="max-w-md truncate text-xs font-medium text-slate-700" title={patent.title}>
              — {patent.title}
            </span>
          )}
          <Badge tone="brand" className="text-2xs font-semibold">
            {chunks.length} {chunks.length === 1 ? 'matching chunk' : 'matching chunks'}
          </Badge>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {patent.best_reranker_score != null ? (
              <>
                <span className="text-xs text-slate-500">Rerank:</span>
                <span className="font-mono text-xs font-bold text-slate-800">
                  {patent.best_reranker_score.toFixed(3)}
                </span>
                {patent.best_retrieval_score != null && (
                  <span
                    className="text-2xs text-slate-400 font-mono"
                    title="Vector similarity score from initial retrieval"
                  >
                    (Vec: {patent.best_retrieval_score.toFixed(3)})
                  </span>
                )}
              </>
            ) : (
              <>
                <span className="text-xs text-slate-500">Sim:</span>
                <span className="font-mono text-xs font-bold text-slate-800">{patent.score.toFixed(3)}</span>
              </>
            )}
            <div className="w-16 h-2 overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full bg-indigo-500 rounded-full"
                style={{ width: `${Math.min(100, Math.max(0, patent.score * 100))}%` }}
              />
            </div>
          </div>

          {/* Dropdown Chevron */}
          <div
            className={`flex size-6 shrink-0 items-center justify-center rounded-md bg-slate-200/70 text-slate-600 transition-transform ${
              isExpanded ? 'rotate-90' : ''
            }`}
            title={isExpanded ? 'Collapse chunks' : 'Expand chunks'}
          >
            <ChevronIcon className="size-3.5" />
          </div>
        </div>
      </div>

      {/* Matching Chunks under this Patent */}
      {isExpanded && (
        <div className="space-y-3 p-3.5 bg-surface sm:p-4">
          {chunks.map((chunk, cIndex) => (
            <ChunkItem
              key={`${chunk.patent_id}-${chunk.chunk_id}-${cIndex}`}
              chunk={chunk}
              chunkIndex={cIndex + 1}
              totalChunks={chunks.length}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ChunkItem({
  chunk,
  chunkIndex,
  totalChunks,
}: {
  chunk: FeatureChunkResult
  chunkIndex: number
  totalChunks: number
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(chunk.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className="rounded-lg border border-slate-200/80 bg-slate-50/60 p-3.5 shadow-2xs">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/60 pb-2">
        <div className="flex flex-wrap items-center gap-2">
          {totalChunks > 1 && (
            <span className="text-2xs font-bold text-slate-700 bg-slate-200/70 rounded px-1.5 py-0.5">
              Chunk {chunkIndex}/{totalChunks}
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
            {chunk.reranker_score != null ? (
              <>
                <span className="text-xs text-slate-500">Rerank:</span>
                <span className="font-mono text-xs font-bold text-slate-800">{chunk.reranker_score.toFixed(3)}</span>
                {chunk.retrieval_score != null && (
                  <span
                    className="text-2xs text-slate-400 font-mono"
                    title="Vector similarity score from initial retrieval"
                  >
                    (Vec: {chunk.retrieval_score.toFixed(3)})
                  </span>
                )}
              </>
            ) : (
              <>
                <span className="text-xs text-slate-500">Sim:</span>
                <span className="font-mono text-xs font-bold text-slate-800">{chunk.score.toFixed(3)}</span>
              </>
            )}
          </div>
          <button
            type="button"
            onClick={handleCopy}
            className="rounded px-2 py-0.5 text-xs font-medium text-slate-500 hover:bg-slate-200 hover:text-slate-800 transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      <div className="mt-2.5 text-xs leading-relaxed text-slate-800 whitespace-pre-wrap font-sans bg-surface p-3.5 rounded-lg border border-slate-200/80">
        <HighlightedText text={chunk.text} limit={chunk.text.length} highlight={chunk.highlight} />
      </div>

      <div className="mt-2 flex items-center justify-between text-2xs text-slate-500">
        <span>Chunk ID: {chunk.chunk_id}</span>
        {chunk.highlight && chunk.highlight.sentences.length > 0 && (
          <span className="text-amber-700 dark:text-amber-300 font-semibold flex items-center gap-1.5">
            <span className="inline-block size-1.5 rounded-full bg-amber-500" />
            Top matching sentence
          </span>
        )}
      </div>
    </div>
  )
}

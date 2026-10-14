import { useMemo, useState } from 'react'
import { Badge, TextField } from '@/components/ui'
import { ChevronIcon, ExternalIcon } from '@/components/ui/icons'
import { HighlightedText } from '@/components/patent/EvidenceChunkCard'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl } from '@/lib/patent'
import type { FeatureChunkResult, KeyFeatureItem, KeyFeatureSearchResponse } from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'

interface StepWiseFilterViewProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse; elapsedMs?: number; error?: string }>
  series: Series[]
}

interface ComprehensivePatentPipelineItem {
  patent_id: string
  title?: string
  featureIds: number[] // Features where this patent is an actual final winning result
  candidateFeatureIds: number[] // All features where this patent was evaluated as a candidate
  bestVectorScore: number
  bestRerankScore: number | null
  vectorRank?: number
  finalRank?: number
  chunksCount: number
  status: 'final' | 'rerank_filtered' | 'topk_pruned' | 'vector_only'
  filterReason?: string
  chunks: FeatureChunkResult[]
  featureBreakdown: {
    feature_id: number
    vector_score?: number
    rerank_score?: number | null
    status: 'final' | 'rerank_filtered' | 'topk_pruned'
    reason?: string
    chunks?: FeatureChunkResult[]
  }[]
}

function scoreTone(score: number): 'success' | 'warning' | 'danger' | 'neutral' {
  if (score >= 0.8) return 'success'
  if (score >= 0.6) return 'warning'
  if (score > 0) return 'danger'
  return 'neutral'
}

function PipelineChunkCard({
  chunk,
  featureId,
  featureText,
}: {
  chunk: FeatureChunkResult
  featureId?: number
  featureText?: string
}) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const body = chunk.text || ''
  const isLong = body.length > 360
  const previewText = expanded || !isLong ? body : body.slice(0, 360)

  const handleCopy = () => {
    navigator.clipboard.writeText(body)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const vScore = chunk.retrieval_score ?? chunk.score
  const rScore = chunk.reranker_score
  const effectiveFeatureId = chunk.feature_id ?? featureId

  return (
    <div className="rounded-lg border border-slate-200/90 dark:border-white/10 bg-surface p-3.5 space-y-2.5 shadow-2xs">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/60 dark:border-white/10 pb-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-2xs font-bold text-slate-800 dark:text-white bg-slate-100 dark:bg-white/10 rounded px-2 py-0.5 border border-slate-200/60 dark:border-white/10">
            Chunk #{chunk.chunk_id}
          </span>
          {effectiveFeatureId != null && (
            <span
              className="font-mono text-2xs font-bold bg-indigo-50 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300 rounded px-1.5 py-0.5 border border-indigo-200 dark:border-indigo-500/30"
              title={featureText ? `Feature ${effectiveFeatureId}: ${featureText}` : `Feature ${effectiveFeatureId}`}
            >
              F{effectiveFeatureId}
            </span>
          )}
          {chunk.section && (
            <span className="text-3xs uppercase font-bold text-slate-600 dark:text-white/70 bg-slate-100 dark:bg-white/10 rounded px-2 py-0.5">
              {chunk.section}
            </span>
          )}
          {chunk.document_chunk_index != null && (
            <span className="text-3xs font-mono text-slate-400 dark:text-white/50">
              Index: {chunk.document_chunk_index}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-2xs font-mono">
            {rScore != null ? (
              <>
                <span className="text-slate-500 dark:text-white/60">Rerank:</span>
                <span className="font-bold text-slate-900 dark:text-white">
                  {rScore.toFixed(3)}
                </span>
                {vScore != null && (
                  <span className="text-slate-400 dark:text-white/50">
                    (Vec: {vScore.toFixed(3)})
                  </span>
                )}
              </>
            ) : (
              <>
                <span className="text-slate-500 dark:text-white/60">Vector:</span>
                <span className="font-bold text-slate-900 dark:text-white">
                  {vScore.toFixed(3)}
                </span>
              </>
            )}
          </div>

          <button
            type="button"
            onClick={handleCopy}
            className="rounded px-2 py-0.5 text-3xs font-medium text-slate-500 hover:text-slate-800 dark:text-white/70 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-white/10 transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      {body ? (
        <div className="text-xs leading-relaxed text-slate-800 dark:text-white/90 whitespace-pre-wrap font-sans bg-slate-50 dark:bg-white/5 p-3 rounded-lg border border-slate-200/70 dark:border-white/10">
          <HighlightedText text={previewText} limit={previewText.length} highlight={chunk.highlight} />
          {isLong && !expanded && '… '}
          {isLong && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="ml-1 text-xs font-semibold text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-300 underline"
            >
              {expanded ? 'Show less' : 'Show full passage'}
            </button>
          )}
        </div>
      ) : (
        <p className="text-2xs text-slate-400 dark:text-white/40 italic">No passage text available.</p>
      )}

      <div className="flex items-center justify-between text-3xs text-slate-400 dark:text-white/50 font-mono">
        <span>Token count: {chunk.token_count ?? '—'}</span>
        {chunk.highlight && chunk.highlight.sentences.length > 0 && (
          <span className="text-amber-700 dark:text-amber-400 font-semibold flex items-center gap-1.5 font-sans">
            <span className="inline-block size-1.5 rounded-full bg-amber-500" />
            Top matching sentence highlighted
          </span>
        )}
      </div>
    </div>
  )
}

export function StepWiseFilterView({ features, runs, series }: StepWiseFilterViewProps) {
  const { data: config } = usePipelineConfig()

  const defaultCollection = series[0]?.name || '512'
  const [selectedCollection, setSelectedCollection] = useState<string>(defaultCollection)
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | 'all'>('all')
  const [statusFilter, setStatusFilter] = useState<'all' | 'final' | 'filtered' | 'pruned'>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedRowId, setExpandedRowId] = useState<string | null>(null)
  const [expandedFeatureFilter, setExpandedFeatureFilter] = useState<number | null>(null)

  const resp = runs[selectedCollection]?.response
  const rerankThreshold = resp?.rerank_score_threshold ?? 0.6
  const topK = resp?.top_k ?? 20

  // Build aggregated step-wise data for the selected collection
  const { patentList, stats } = useMemo(() => {
    if (!resp || !resp.results) {
      return {
        patentList: [] as ComprehensivePatentPipelineItem[],
        stats: { vectorTotal: 0, rerankPassedTotal: 0, rerankFilteredTotal: 0, topkPrunedTotal: 0, finalTotal: 0 },
      }
    }

    const map = new Map<string, ComprehensivePatentPipelineItem>()

    const getOrCreateItem = (
      patentId: string,
      title: string | undefined | null,
      vScore: number,
      rScore: number | null,
      initialStatus: 'final' | 'rerank_filtered' | 'topk_pruned' | 'vector_only',
      rank?: number,
      chunksCount: number = 1,
      reason?: string,
      initialChunks: FeatureChunkResult[] = []
    ): ComprehensivePatentPipelineItem => {
      let existing = map.get(patentId)
      if (!existing) {
        existing = {
          patent_id: patentId,
          title: title || undefined,
          featureIds: [],
          candidateFeatureIds: [],
          bestVectorScore: vScore,
          bestRerankScore: rScore,
          finalRank: rank,
          chunksCount,
          status: initialStatus,
          filterReason: reason,
          chunks: [...initialChunks],
          featureBreakdown: [],
        }
        map.set(patentId, existing)
      } else {
        for (const ch of initialChunks) {
          const chFid = ch.feature_id ?? (ch.metadata?.feature_id as number | undefined)
          const exists = existing.chunks.some((c) => {
            const cFid = c.feature_id ?? (c.metadata?.feature_id as number | undefined)
            return c.chunk_id === ch.chunk_id && (chFid == null || cFid == null || cFid === chFid)
          })
          if (!exists) {
            existing.chunks.push(ch)
          }
        }
      }
      return existing
    }

    // Filter results if a specific feature is selected
    const activeResults =
      selectedFeatureId === 'all'
        ? resp.results
        : resp.results.filter((r) => r.feature_id === selectedFeatureId)

    for (const featRes of activeResults) {
      const featId = featRes.feature_id

      // 1. Process winning final patents
      for (let idx = 0; idx < (featRes.patents || []).length; idx++) {
        const p = featRes.patents[idx]
        const vScore = p.best_retrieval_score ?? 0.0
        const rScore = p.best_reranker_score ?? p.score
        const taggedChunks: FeatureChunkResult[] = (p.chunks || []).map((ch) => ({
          ...ch,
          feature_id: ch.feature_id ?? featId,
        }))
        const item = getOrCreateItem(
          p.patent_id,
          p.title,
          vScore,
          rScore,
          'final',
          idx + 1,
          taggedChunks.length || 1,
          undefined,
          taggedChunks
        )

        if (!item.title && p.title) item.title = p.title
        if (vScore > item.bestVectorScore) item.bestVectorScore = vScore
        if (item.bestRerankScore === null || rScore > item.bestRerankScore) {
          item.bestRerankScore = rScore
        }
        item.status = 'final'

        if (!item.candidateFeatureIds.includes(featId)) item.candidateFeatureIds.push(featId)
        if (!item.featureIds.includes(featId)) item.featureIds.push(featId)

        // Ensure winning chunks are merged into item.chunks
        for (const ch of taggedChunks) {
          const chFid = ch.feature_id ?? featId
          const exists = item.chunks.some((c) => {
            const cFid = c.feature_id ?? (c.metadata?.feature_id as number | undefined)
            return c.chunk_id === ch.chunk_id && cFid === chFid
          })
          if (!exists) {
            item.chunks.push(ch)
          }
        }

        item.featureBreakdown.push({
          feature_id: featId,
          vector_score: vScore,
          rerank_score: rScore,
          status: 'final',
          chunks: taggedChunks,
        })
      }

      // 2. Process patents pruned by top-K cutoff
      for (const p of featRes.topk_pruned_patents || []) {
        const vScore = p.retrieval_score ?? 0.0
        const rScore = p.reranker_score ?? null
        const taggedChunks: FeatureChunkResult[] = (p.chunks || []).map((ch) => ({
          ...ch,
          feature_id: ch.feature_id ?? featId,
        }))
        const item = getOrCreateItem(
          p.patent_id,
          p.title,
          vScore,
          rScore,
          'topk_pruned',
          p.rank || undefined,
          p.chunks_count || taggedChunks.length || 1,
          p.filter_reason || `Rank #${p.rank} > Top-${topK}`,
          taggedChunks
        )

        if (!item.title && p.title) item.title = p.title
        if (vScore > item.bestVectorScore) item.bestVectorScore = vScore
        if (rScore !== null && (item.bestRerankScore === null || rScore > item.bestRerankScore)) {
          item.bestRerankScore = rScore
        }
        if (item.status !== 'final') {
          item.status = 'topk_pruned'
          item.filterReason = p.filter_reason || `Rank #${p.rank} > Top-${topK}`
        }
        if (!item.candidateFeatureIds.includes(featId)) item.candidateFeatureIds.push(featId)

        item.featureBreakdown.push({
          feature_id: featId,
          vector_score: vScore,
          rerank_score: rScore,
          status: 'topk_pruned',
          reason: p.filter_reason || undefined,
          chunks: taggedChunks,
        })
      }

      // 3. Process patents filtered by rerank score threshold
      for (const p of featRes.rerank_filtered_patents || []) {
        const vScore = p.retrieval_score ?? 0.0
        const rScore = p.reranker_score ?? null
        const taggedChunks: FeatureChunkResult[] = (p.chunks || []).map((ch) => ({
          ...ch,
          feature_id: ch.feature_id ?? featId,
        }))
        const item = getOrCreateItem(
          p.patent_id,
          p.title,
          vScore,
          rScore,
          'rerank_filtered',
          undefined,
          p.chunks_count || taggedChunks.length || 1,
          p.filter_reason || `Score below threshold ${rerankThreshold.toFixed(2)}`,
          taggedChunks
        )

        if (!item.title && p.title) item.title = p.title
        if (vScore > item.bestVectorScore) item.bestVectorScore = vScore
        if (rScore !== null && (item.bestRerankScore === null || rScore > item.bestRerankScore)) {
          item.bestRerankScore = rScore
        }
        if (item.status !== 'final' && item.status !== 'topk_pruned') {
          item.status = 'rerank_filtered'
          item.filterReason = p.filter_reason || `Score below threshold ${rerankThreshold.toFixed(2)}`
        }
        if (!item.candidateFeatureIds.includes(featId)) item.candidateFeatureIds.push(featId)

        item.featureBreakdown.push({
          feature_id: featId,
          vector_score: vScore,
          rerank_score: rScore,
          status: 'rerank_filtered',
          reason: p.filter_reason || undefined,
          chunks: taggedChunks,
        })
      }

      // 4. Process all vector retrieval patents (ensures complete candidate coverage)
      for (const p of featRes.vector_retrieval_patents || []) {
        const vScore = p.retrieval_score ?? 0.0
        const taggedChunks: FeatureChunkResult[] = (p.chunks || []).map((ch) => ({
          ...ch,
          feature_id: ch.feature_id ?? featId,
        }))
        const item = getOrCreateItem(
          p.patent_id,
          p.title,
          vScore,
          null,
          'rerank_filtered',
          p.rank || undefined,
          p.chunks_count || taggedChunks.length || 1,
          'Filtered during reranking',
          taggedChunks
        )

        if (!item.title && p.title) item.title = p.title
        if (vScore > item.bestVectorScore) item.bestVectorScore = vScore
        if (!item.candidateFeatureIds.includes(featId)) item.candidateFeatureIds.push(featId)
      }

      // 5. Global fallback from featRes.chunks
      for (const ch of featRes.chunks || []) {
        const it = map.get(ch.patent_id)
        if (it) {
          const tagged = { ...ch, feature_id: ch.feature_id ?? featId }
          const exists = it.chunks.some((c) => {
            const cFid = c.feature_id ?? (c.metadata?.feature_id as number | undefined)
            return c.chunk_id === ch.chunk_id && (cFid === featId || cFid == null)
          })
          if (!exists) {
            it.chunks.push(tagged)
          }
        }
      }
    }

    const items = Array.from(map.values())

    for (const item of items) {
      item.featureIds.sort((a, b) => a - b)
      item.candidateFeatureIds.sort((a, b) => a - b)

      // When patent is a winning result ('final') and won one or more features:
      // Retain all matching chunks from winning features
      if (item.status === 'final' && item.featureIds.length > 0) {
        const winningChunks = item.chunks.filter((c) => {
          const fid = c.feature_id ?? (c.metadata?.feature_id as number | undefined)
          return fid != null && item.featureIds.includes(fid)
        })
        if (winningChunks.length > 0) {
          item.chunks = winningChunks
        }
      }

      // Sort matching chunks:
      // 1. By feature_id ascending (F1, then F3, then F9)
      // 2. By reranker_score / retrieval score descending
      item.chunks.sort((a, b) => {
        const featA = a.feature_id ?? (a.metadata?.feature_id as number | undefined) ?? 0
        const featB = b.feature_id ?? (b.metadata?.feature_id as number | undefined) ?? 0
        if (featA !== featB) return featA - featB
        const scoreA = a.reranker_score ?? a.retrieval_score ?? a.score
        const scoreB = b.reranker_score ?? b.retrieval_score ?? b.score
        return scoreB - scoreA
      })
    }

    // Compute funnel counts
    const vectorTotal = items.length
    const finalTotal = items.filter((p) => p.status === 'final').length
    const topkPrunedTotal = items.filter((p) => p.status === 'topk_pruned').length
    const rerankFilteredTotal = items.filter((p) => p.status === 'rerank_filtered').length
    const rerankPassedTotal = finalTotal + topkPrunedTotal

    // Sort: 1. Final winners first, 2. Top-K pruned, 3. Filtered, then by best score
    items.sort((a, b) => {
      const order = { final: 0, topk_pruned: 1, rerank_filtered: 2, vector_only: 3 }
      if (order[a.status] !== order[b.status]) {
        return order[a.status] - order[b.status]
      }
      const scoreA = a.bestRerankScore ?? a.bestVectorScore
      const scoreB = b.bestRerankScore ?? b.bestVectorScore
      return scoreB - scoreA
    })

    return {
      patentList: items,
      stats: { vectorTotal, rerankPassedTotal, rerankFilteredTotal, topkPrunedTotal, finalTotal },
    }
  }, [resp, selectedFeatureId, rerankThreshold, topK])

  // Filter list by status & search query
  const filteredList = useMemo(() => {
    return patentList.filter((p) => {
      if (statusFilter === 'final' && p.status !== 'final') return false
      if (statusFilter === 'filtered' && p.status !== 'rerank_filtered') return false
      if (statusFilter === 'pruned' && p.status !== 'topk_pruned') return false

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim()
        const matchId = p.patent_id.toLowerCase().includes(q)
        const matchTitle = p.title?.toLowerCase().includes(q)
        const matchFeat =
          p.featureIds.some((id) => `f${id}`.includes(q)) ||
          p.candidateFeatureIds.some((id) => `f${id}`.includes(q))
        if (!matchId && !matchTitle && !matchFeat) return false
      }
      return true
    })
  }, [patentList, statusFilter, searchQuery])

  const toggleExpand = (patentId: string) => {
    setExpandedFeatureFilter(null)
    setExpandedRowId((prev) => (prev === patentId ? null : patentId))
  }

  const selectedSeries = series.find((s) => s.name === selectedCollection)

  return (
    <div className="space-y-6">
      {/* Top Banner & Pipeline Explanation */}
      <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs p-4 sm:p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Pipeline Step-Wise Patent Filtering
            </h3>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-white/70">
              Shows every candidate patent retrieved from Qdrant vector search and tracks which step filtered or selected each patent.
            </p>
          </div>

          {/* Collection Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-2xs font-bold uppercase tracking-wider text-slate-400 dark:text-white/60">
              Collection:
            </span>
            <div className="flex items-center gap-1">
              {series.map((s) => (
                <button
                  key={s.name}
                  type="button"
                  onClick={() => setSelectedCollection(s.name)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                    selectedCollection === s.name
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white hover:bg-slate-200 dark:hover:bg-white/20'
                  }`}
                >
                  <span className="size-2 rounded-xs" style={{ backgroundColor: s.color }} />
                  <span>{s.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 4-Step Visual Filter Funnel */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-2">
          {/* Step 1: Vector Retrieval */}
          <div className="rounded-lg border border-slate-200 dark:border-white/10 bg-slate-50/70 dark:bg-white/5 p-3 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-2xs font-bold uppercase tracking-wider text-slate-500 dark:text-white/70">
                1. Vector Retrieval
              </span>
              <span className="px-1.5 py-0.5 rounded text-3xs font-bold font-mono bg-slate-200 dark:bg-white/10 text-slate-800 dark:text-white">
                Initial
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-slate-900 dark:text-white">
              {stats.vectorTotal}
            </div>
            <p className="text-3xs text-slate-500 dark:text-white/60">
              Candidates retrieved from Qdrant by cosine similarity
            </p>
          </div>

          {/* Step 2: Rerank Score Filter */}
          <div className="rounded-lg border border-amber-300/80 dark:border-amber-500/30 bg-amber-50/50 dark:bg-amber-500/10 p-3 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-2xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-400">
                2. Cross-Encoder Rerank
              </span>
              <span className="px-1.5 py-0.5 rounded text-3xs font-bold font-mono bg-amber-100 text-amber-900 dark:bg-amber-500/25 dark:text-amber-300 dark:ring-1 dark:ring-amber-400/30">
                ≥ {rerankThreshold.toFixed(2)}
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-slate-900 dark:text-white">
                {stats.rerankPassedTotal}
              </span>
              <span className="text-2xs font-mono text-rose-600 dark:text-rose-400 font-medium">
                ({stats.rerankFilteredTotal} filtered)
              </span>
            </div>
            <p className="text-3xs text-slate-500 dark:text-white/70">
              Filtered if best chunk &lt; threshold {rerankThreshold.toFixed(2)}
            </p>
          </div>

          {/* Step 3: Top-K Cutoff */}
          <div className="rounded-lg border border-indigo-200 dark:border-indigo-500/30 bg-indigo-50/50 dark:bg-indigo-500/10 p-3 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-2xs font-bold uppercase tracking-wider text-indigo-800 dark:text-indigo-400">
                3. Top-K Cutoff
              </span>
              <span className="px-1.5 py-0.5 rounded text-3xs font-bold font-mono bg-indigo-100 text-indigo-900 dark:bg-indigo-500/25 dark:text-indigo-300 dark:ring-1 dark:ring-indigo-400/30">
                Top {topK}
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-slate-900 dark:text-white">
                {stats.finalTotal}
              </span>
              {stats.topkPrunedTotal > 0 && (
                <span className="text-2xs font-mono text-slate-500 dark:text-white/70 font-medium">
                  ({stats.topkPrunedTotal} pruned)
                </span>
              )}
            </div>
            <p className="text-3xs text-slate-500 dark:text-white/70">
              Only highest-scoring {topK} patents retained per feature
            </p>
          </div>

          {/* Step 4: Final Winning Patents */}
          <div className="rounded-lg border border-emerald-300 dark:border-emerald-500/30 bg-emerald-50/50 dark:bg-emerald-500/10 p-3 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-2xs font-bold uppercase tracking-wider text-emerald-800 dark:text-emerald-400">
                4. Final Winners
              </span>
              <span className="px-1.5 py-0.5 rounded text-3xs font-bold font-mono bg-emerald-100 text-emerald-900 dark:bg-emerald-500/25 dark:text-emerald-300 dark:ring-1 dark:ring-emerald-400/30">
                ✓ Passed
              </span>
            </div>
            <div className="text-2xl font-bold font-mono text-slate-900 dark:text-white">
              {stats.finalTotal}
            </div>
            <p className="text-3xs text-slate-500 dark:text-white/70">
              Surviving patents enriched with full metadata & passages
            </p>
          </div>
        </div>

        {/* Feature Pills Switcher */}
        <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-200 dark:border-white/10">
          <span className="text-2xs font-bold uppercase tracking-wider text-slate-400 dark:text-white/60 mr-1">
            Feature:
          </span>
          <button
            type="button"
            onClick={() => setSelectedFeatureId('all')}
            className={`px-2.5 py-1 rounded text-2xs font-bold transition-all ${
              selectedFeatureId === 'all'
                ? 'bg-indigo-600 text-white shadow-2xs'
                : 'bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white hover:bg-slate-200 dark:hover:bg-white/20'
            }`}
          >
            All Features ({features.length})
          </button>
          {features.map((f) => (
            <button
              key={f.feature_id}
              type="button"
              onClick={() => setSelectedFeatureId(f.feature_id)}
              className={`px-2 py-1 rounded font-mono text-2xs font-bold transition-all ${
                selectedFeatureId === f.feature_id
                  ? 'bg-indigo-600 text-white shadow-2xs'
                  : 'bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white hover:bg-slate-200 dark:hover:bg-white/20'
              }`}
              title={f.feature}
            >
              F{f.feature_id}
            </button>
          ))}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 dark:border-white/10 bg-surface p-3 shadow-xs">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-2xs font-bold uppercase tracking-wider text-slate-400 dark:text-white/60">
            Show:
          </span>
          <button
            type="button"
            onClick={() => setStatusFilter('all')}
            className={`px-2.5 py-1 rounded text-2xs font-medium transition-all ${
              statusFilter === 'all'
                ? 'bg-indigo-600 text-white font-bold shadow-2xs'
                : 'bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white hover:bg-slate-200 dark:hover:bg-white/20'
            }`}
          >
            All Candidates ({patentList.length})
          </button>
          <button
            type="button"
            onClick={() => setStatusFilter('final')}
            className={`px-2.5 py-1 rounded text-2xs font-medium transition-all ${
              statusFilter === 'final'
                ? 'bg-emerald-600 text-white font-bold shadow-2xs'
                : 'bg-emerald-50 text-emerald-800 border border-emerald-200/80 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30 hover:bg-emerald-100 dark:hover:bg-emerald-500/25'
            }`}
          >
            ✓ Final Winners ({stats.finalTotal})
          </button>
          <button
            type="button"
            onClick={() => setStatusFilter('filtered')}
            className={`px-2.5 py-1 rounded text-2xs font-medium transition-all ${
              statusFilter === 'filtered'
                ? 'bg-amber-600 text-white font-bold shadow-2xs'
                : 'bg-amber-50 text-amber-900 border border-amber-200/80 dark:bg-amber-500/15 dark:text-amber-300 dark:border-amber-500/30 hover:bg-amber-100 dark:hover:bg-amber-500/25'
            }`}
          >
            ✗ Filtered by Threshold ({stats.rerankFilteredTotal})
          </button>
          {stats.topkPrunedTotal > 0 && (
            <button
              type="button"
              onClick={() => setStatusFilter('pruned')}
              className={`px-2.5 py-1 rounded text-2xs font-medium transition-all ${
                statusFilter === 'pruned'
                  ? 'bg-indigo-600 text-white font-bold shadow-2xs'
                  : 'bg-indigo-50 text-indigo-800 border border-indigo-200/80 dark:bg-indigo-500/15 dark:text-indigo-300 dark:border-indigo-500/30 hover:bg-indigo-100 dark:hover:bg-indigo-500/25'
              }`}
            >
              ✗ Pruned by Top-K ({stats.topkPrunedTotal})
            </button>
          )}
        </div>

        {/* Quick Search */}
        <div className="w-full sm:w-64">
          <TextField
            placeholder="Search patent ID or title..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="text-xs"
          />
        </div>
      </div>

      {/* Patent Journey Table */}
      <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-surface shadow-xs overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 px-4 py-3">
          <div className="flex items-center gap-2">
            {selectedSeries && (
              <span className="size-3 rounded-xs shrink-0" style={{ backgroundColor: selectedSeries.color }} />
            )}
            <h4 className="font-bold text-sm text-slate-900 dark:text-white">
              Collection “{selectedCollection}” — Candidate Patents Flow
            </h4>
          </div>
          <span className="font-mono text-2xs text-slate-500 dark:text-white/70">
            Showing {filteredList.length} of {patentList.length} candidates
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 dark:divide-white/10 text-xs">
            <thead className="bg-slate-50 dark:bg-surface">
              <tr>
                <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs w-12">
                  #
                </th>
                <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs min-w-[200px]">
                  Patent ID & Title
                </th>
                <th
                  className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs"
                  title="Features where this patent was selected as a final winning result"
                >
                  Final Features
                </th>
                <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                  Vector Retrieval
                </th>
                <th className="px-3.5 py-2.5 text-center font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                  Rerank Score
                </th>
                <th className="px-3.5 py-2.5 text-left font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs">
                  Pipeline Step Outcome
                </th>
                <th className="px-3.5 py-2.5 text-right font-semibold text-slate-500 dark:text-white/70 uppercase text-2xs w-20">
                  Inspect
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-white/10 bg-surface">
              {filteredList.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-8 text-center text-xs text-slate-400 dark:text-white/40 italic">
                    No patents match the current filter in “{selectedCollection}”.
                  </td>
                </tr>
              ) : (
                filteredList.map((item, idx) => {
                  const url = patentUrl(config?.patent_view_url_template, item.patent_id)
                  const isExpanded = expandedRowId === item.patent_id

                  return (
                    <>
                      <tr
                        key={item.patent_id}
                        className={`hover:bg-slate-50/70 dark:hover:bg-white/5 transition-colors ${
                          item.status === 'final'
                            ? 'bg-emerald-50/20 dark:bg-emerald-950/10'
                            : item.status === 'rerank_filtered'
                            ? 'opacity-85'
                            : ''
                        }`}
                      >
                        {/* Rank */}
                        <td className="px-3.5 py-3 align-middle text-center">
                          <span className="font-mono text-xs font-bold text-slate-700 dark:text-white">
                            #{idx + 1}
                          </span>
                        </td>

                        {/* Patent ID & Title */}
                        <td className="px-3.5 py-3 align-middle">
                          <div className="flex flex-col">
                            {url ? (
                              <a
                                href={url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="font-mono text-xs font-bold text-indigo-600 hover:underline dark:text-white inline-flex items-center gap-1"
                              >
                                <span>{item.patent_id}</span>
                                <ExternalIcon className="size-2.5 opacity-60" />
                              </a>
                            ) : (
                              <span className="font-mono text-xs font-bold text-slate-900 dark:text-white">
                                {item.patent_id}
                              </span>
                            )}
                            {item.title && (
                              <span
                                className="text-2xs text-slate-500 dark:text-white/70 truncate max-w-sm mt-0.5"
                                title={item.title}
                              >
                                {item.title}
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Features Matched (Final Winning Results) */}
                        <td className="px-3.5 py-3 align-middle">
                          <div className="flex flex-wrap items-center gap-1">
                            {item.featureIds.length > 0 ? (
                              item.featureIds.map((fid) => (
                                <span
                                  key={fid}
                                  className="px-1.5 py-0.5 rounded font-mono text-2xs font-bold bg-indigo-50 dark:bg-white/10 text-indigo-700 dark:text-white border border-indigo-200 dark:border-white/15"
                                  title={`Final winning result for Feature ${fid}`}
                                >
                                  F{fid}
                                </span>
                              ))
                            ) : (
                              <span
                                className="text-2xs text-slate-400 dark:text-white/40 italic font-mono"
                                title={
                                  item.candidateFeatureIds.length > 0
                                    ? `Candidate in F${item.candidateFeatureIds.join(', F')}, but not selected in final top-K`
                                    : 'No final features'
                                }
                              >
                                —
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Vector Retrieval */}
                        <td className="px-3.5 py-3 align-middle text-center">
                          <div className="flex flex-col items-center">
                            <span className="font-mono text-xs font-bold text-slate-800 dark:text-white">
                              {item.bestVectorScore.toFixed(3)}
                            </span>
                            <span className="text-3xs text-slate-400 dark:text-white/50">
                              ({item.chunksCount} chk)
                            </span>
                          </div>
                        </td>

                        {/* Rerank Score */}
                        <td className="px-3.5 py-3 align-middle text-center">
                          {item.bestRerankScore !== null ? (
                            <Badge
                              tone={scoreTone(item.bestRerankScore)}
                              className="font-mono text-2xs font-bold dark:text-white"
                            >
                              {item.bestRerankScore.toFixed(3)}
                            </Badge>
                          ) : (
                            <span className="text-slate-400 dark:text-white/40 italic text-2xs">—</span>
                          )}
                        </td>

                        {/* Step Outcome */}
                        <td className="px-3.5 py-3 align-middle">
                          {item.status === 'final' ? (
                            <div className="flex items-center gap-1.5">
                              <span className="flex size-4 items-center justify-center rounded-full bg-emerald-600 text-white text-3xs font-bold">
                                ✓
                              </span>
                              <span className="font-semibold text-emerald-700 dark:text-emerald-400 text-2xs">
                                Winning Result (Rank #{item.finalRank || 1} in Top-{topK})
                              </span>
                            </div>
                          ) : item.status === 'topk_pruned' ? (
                            <div className="flex items-center gap-1.5">
                              <span className="flex size-4 items-center justify-center rounded-full bg-indigo-600 text-white text-3xs font-bold">
                                ✂
                              </span>
                              <span className="font-medium text-indigo-700 dark:text-indigo-400 text-2xs">
                                Pruned: Passed rerank, but rank exceeds Top-{topK} limit
                              </span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1.5">
                              <span className="flex size-4 items-center justify-center rounded-full bg-rose-600 text-white text-3xs font-bold">
                                ✕
                              </span>
                              <span className="font-medium text-rose-700 dark:text-rose-400 text-2xs">
                                Filtered: {item.filterReason || `Rerank score < ${rerankThreshold.toFixed(2)}`}
                              </span>
                            </div>
                          )}
                        </td>

                        {/* Inspect Details */}
                        <td className="px-3.5 py-3 align-middle text-right">
                          <button
                            type="button"
                            onClick={() => toggleExpand(item.patent_id)}
                            className="inline-flex items-center gap-1 text-2xs font-medium text-indigo-600 hover:text-indigo-800 dark:text-indigo-300 dark:hover:text-white"
                          >
                            <span>{isExpanded ? 'Hide' : 'Trace'}</span>
                            <ChevronIcon
                              className={`size-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                            />
                          </button>
                        </td>
                      </tr>

                      {/* Expanded Breakdown by Feature & Matching Chunks */}
                      {isExpanded && (() => {
                        const displayChunks =
                          expandedFeatureFilter === null
                            ? item.chunks
                            : item.chunks.filter(
                                (c) =>
                                  (c.feature_id ?? (c.metadata?.feature_id as number | undefined)) ===
                                  expandedFeatureFilter
                              ).length > 0
                            ? item.chunks.filter(
                                (c) =>
                                  (c.feature_id ?? (c.metadata?.feature_id as number | undefined)) ===
                                  expandedFeatureFilter
                              )
                            : item.featureBreakdown.find((fb) => fb.feature_id === expandedFeatureFilter)
                                ?.chunks || []

                        const availablePills =
                          item.featureIds.length > 0 ? item.featureIds : item.candidateFeatureIds

                        return (
                          <tr className="bg-slate-50/80 dark:bg-white/5">
                            <td colSpan={7} className="px-6 py-4">
                              <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                  <div className="text-2xs font-bold uppercase tracking-wider text-slate-500 dark:text-white/70">
                                    Feature-by-Feature Pipeline Filtering for {item.patent_id}
                                  </div>
                                  <span className="font-mono text-2xs text-slate-500 dark:text-white/70">
                                    {item.chunks.length} matching chunk{item.chunks.length === 1 ? '' : 's'}
                                  </span>
                                </div>

                                {/* Feature Cards Grid */}
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                  {item.featureBreakdown.map((fb, fbIdx) => {
                                    const featObj = features.find((f) => f.feature_id === fb.feature_id)
                                    const isSelectedFeat = expandedFeatureFilter === fb.feature_id
                                    return (
                                      <div
                                        key={fbIdx}
                                        onClick={() =>
                                          setExpandedFeatureFilter((prev) =>
                                            prev === fb.feature_id ? null : fb.feature_id
                                          )
                                        }
                                        className={`rounded-lg border p-3 space-y-2 shadow-2xs cursor-pointer transition-all ${
                                          isSelectedFeat
                                            ? 'border-indigo-500 ring-2 ring-indigo-500/30 bg-indigo-50/30 dark:bg-indigo-950/30'
                                            : 'border-slate-200 dark:border-white/10 bg-surface hover:border-slate-300 dark:hover:border-white/20'
                                        }`}
                                        title={`Click to filter matching passages to Feature ${fb.feature_id}`}
                                      >
                                        <div className="flex items-center justify-between border-b border-slate-200 dark:border-white/10 pb-1.5">
                                          <span className="font-mono font-bold text-xs text-indigo-600 dark:text-indigo-300">
                                            Feature {fb.feature_id}
                                          </span>
                                          <Badge
                                            tone={
                                              fb.status === 'final'
                                                ? 'success'
                                                : fb.status === 'topk_pruned'
                                                ? 'brand'
                                                : 'danger'
                                            }
                                            className="text-3xs font-bold dark:text-white"
                                          >
                                            {fb.status === 'final'
                                              ? 'Passed'
                                              : fb.status === 'topk_pruned'
                                              ? 'Top-K Cutoff'
                                              : 'Threshold Filtered'}
                                          </Badge>
                                        </div>

                                        {featObj && (
                                          <p className="text-2xs text-slate-600 dark:text-white/80 line-clamp-2">
                                            {featObj.feature}
                                          </p>
                                        )}

                                        <div className="pt-1 border-t border-slate-200 dark:border-white/10 flex items-center justify-between text-2xs font-mono">
                                          <span className="text-slate-500 dark:text-white/60">
                                            Vector: {fb.vector_score ? fb.vector_score.toFixed(3) : '—'}
                                          </span>
                                          <span className="font-bold text-slate-900 dark:text-white">
                                            Rerank: {fb.rerank_score != null ? fb.rerank_score.toFixed(3) : '—'}
                                          </span>
                                        </div>

                                        {fb.reason && (
                                          <p className="text-3xs text-rose-600 dark:text-rose-400 italic">
                                            {fb.reason}
                                          </p>
                                        )}
                                      </div>
                                    )
                                  })}
                                </div>

                                {/* Matching Chunks Section */}
                                <div className="space-y-3 pt-3 border-t border-slate-200 dark:border-white/10">
                                  <div className="flex flex-wrap items-center justify-between gap-2">
                                    <div className="flex items-center gap-2">
                                      <span className="text-2xs font-bold uppercase tracking-wider text-slate-700 dark:text-white">
                                        Matching Chunks ({displayChunks.length})
                                      </span>
                                      <span className="text-3xs text-slate-400 dark:text-white/50">
                                        Candidate text passages retrieved & scored for this patent
                                      </span>
                                    </div>

                                    {/* Feature Filter Pills */}
                                    {availablePills.length > 1 && (
                                      <div className="flex flex-wrap items-center gap-1.5">
                                        <span className="text-3xs text-slate-500 dark:text-white/60 mr-0.5">
                                          Filter chunks:
                                        </span>
                                        <button
                                          type="button"
                                          onClick={() => setExpandedFeatureFilter(null)}
                                          className={`px-2 py-0.5 rounded text-2xs font-semibold transition-colors ${
                                            expandedFeatureFilter === null
                                              ? 'bg-indigo-600 text-white shadow-2xs'
                                              : 'bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-white/10 dark:text-white/80 dark:hover:bg-white/20'
                                          }`}
                                        >
                                          All ({item.chunks.length})
                                        </button>
                                        {availablePills.map((fid) => {
                                          const count = item.chunks.filter(
                                            (c) =>
                                              (c.feature_id ??
                                                (c.metadata?.feature_id as number | undefined)) === fid
                                          ).length
                                          return (
                                            <button
                                              key={fid}
                                              type="button"
                                              onClick={() => setExpandedFeatureFilter(fid)}
                                              className={`px-2 py-0.5 rounded text-2xs font-mono font-bold transition-colors ${
                                                expandedFeatureFilter === fid
                                                  ? 'bg-indigo-600 text-white shadow-2xs'
                                                  : 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-white/10 dark:text-white dark:hover:bg-white/20 border border-indigo-200 dark:border-white/10'
                                              }`}
                                            >
                                              F{fid} ({count})
                                            </button>
                                          )
                                        })}
                                      </div>
                                    )}
                                  </div>

                                  {displayChunks.length > 0 ? (
                                    <div className="space-y-3">
                                      {displayChunks.map((chunk, cIdx) => {
                                        const cFeatId =
                                          chunk.feature_id ??
                                          (chunk.metadata?.feature_id as number | undefined) ??
                                          item.featureIds[0] ??
                                          item.candidateFeatureIds[0]
                                        const featText = features.find(
                                          (f) => f.feature_id === cFeatId
                                        )?.feature

                                        return (
                                          <PipelineChunkCard
                                            key={`${chunk.patent_id}-${chunk.chunk_id}-${cFeatId}-${cIdx}`}
                                            chunk={chunk}
                                            featureId={cFeatId}
                                            featureText={featText}
                                          />
                                        )
                                      })}
                                    </div>
                                  ) : (
                                    <div className="rounded-lg border border-slate-200/80 dark:border-white/10 bg-surface p-4 text-center text-xs text-slate-400 dark:text-white/50 italic">
                                      No chunk text passages available for {item.patent_id}
                                      {expandedFeatureFilter ? ` in Feature ${expandedFeatureFilter}` : ''}.
                                    </div>
                                  )}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )
                      })()}
                    </>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

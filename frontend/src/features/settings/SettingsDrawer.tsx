import { useCallback } from 'react'
import { Alert, Button, Drawer, KeyValueList, ScoreBar, Section, Spinner, Stat, StatGrid } from '@/components/ui'
import { useCacheStats, useClearCache, useCollections, usePipelineConfig } from '@/hooks/queries'
import { formatPercent } from '@/lib/format'
import { useAppDispatch, useAppSelector } from '@/store'
import { settingsToggled } from '@/store/uiSlice'
import { AppearanceSettings } from './AppearanceSettings'

const WEIGHT_LABELS: Record<string, string> = {
  relationship: 'Relationships',
  requirement: 'Requirements',
  reranker: 'Cross-encoder reranker',
  retrieval: 'Vector retrieval',
}

export function SettingsDrawer() {
  const dispatch = useAppDispatch()
  const open = useAppSelector((s) => s.ui.settingsOpen)
  const onClose = useCallback(() => dispatch(settingsToggled(false)), [dispatch])

  return (
    <Drawer open={open} onClose={onClose} title="Settings" subtitle="Appearance, cache and server configuration">
      <div className="space-y-8">
        <AppearanceSettings />
        <CacheSection />
        <CollectionsSection />
        <ConfigSection />
      </div>
    </Drawer>
  )
}

function CacheSection() {
  const { data: stats, isPending, error } = useCacheStats()
  const clearCache = useClearCache()

  return (
    <Section
      title="Query understanding cache"
      aside={
        <Button variant="danger" size="sm" loading={clearCache.isPending} onClick={() => clearCache.mutate()}>
          Clear cache
        </Button>
      }
    >
      {isPending && <Spinner className="size-5 text-slate-400" />}
      {error && <Alert tone="danger">{error.message}</Alert>}
      {clearCache.error && <Alert tone="danger">{clearCache.error.message}</Alert>}
      {stats && (
        <StatGrid>
          <Stat label="Cached" value={`${stats.size} / ${stats.max_size}`} />
          <Stat label="Hit ratio" value={formatPercent(stats.hit_ratio, 1)} />
          <Stat label="Hits" value={stats.hits} />
          <Stat label="Misses" value={stats.misses} />
        </StatGrid>
      )}
    </Section>
  )
}

function CollectionsSection() {
  const { data, isPending, error } = useCollections()

  if (isPending) return <Spinner className="size-5 text-slate-400" />
  if (error) return <Alert tone="danger">{error.message}</Alert>

  return (
    <Section title="Qdrant collections">
      {data.collections.length === 0 ? (
        <p className="text-sm text-slate-500">
          No searchable collection. Search needs a patent_chunks_&lt;name&gt; and patents_metadata_&lt;name&gt; pair.
        </p>
      ) : (
        <KeyValueList
          items={data.collections.map((c) => ({
            label: c.name === data.default ? `${c.name} (default)` : c.name,
            value: `${c.chunk_count.toLocaleString()} chunks · ${c.patent_count.toLocaleString()} patents`,
          }))}
        />
      )}
    </Section>
  )
}

function ConfigSection() {
  const { data: config, isPending, error } = usePipelineConfig()

  if (isPending) return <Spinner className="size-5 text-slate-400" />
  if (error) return <Alert tone="danger">{error.message}</Alert>

  return (
    <>
      <Section title={`Final score weights · threshold ${config.final_score_threshold} / 10`}>
        <div className="space-y-3">
          {Object.entries(config.weights).map(([key, weight]) => (
            <ScoreBar key={key} label={WEIGHT_LABELS[key] ?? key} value={weight} display={formatPercent(weight)} />
          ))}
        </div>
      </Section>

      <Section title="Models & infrastructure">
        <KeyValueList
          items={[
            { label: 'Query LLM', value: config.query_llm_model },
            { label: 'LLM base URL', value: config.query_llm_base_url },
            { label: 'Embedding model', value: config.embedding_model },
            { label: 'Verifier / reranker', value: config.reranker_model },
          ]}
        />
      </Section>

      <Section title="Retrieval limits">
        <KeyValueList
          items={[
            { label: 'Top K per view', value: config.retrieval_top_k_per_view },
            { label: 'Candidate limit', value: config.patent_candidate_top_k },
            { label: 'Neighbor radius', value: `±${config.evidence_neighbor_chunks} chunks` },
            { label: 'Rerank batch size', value: config.rerank_batch_size },
            { label: 'Rerank max tokens', value: config.reranker_max_context_tokens },
          ]}
        />
      </Section>
    </>
  )
}

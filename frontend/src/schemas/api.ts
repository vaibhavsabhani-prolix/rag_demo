/** Zod schemas for the FastAPI endpoints (app/api/schemas.py) and the search stream. */
import { z } from 'zod'

export const pipelineConfigSchema = z.object({
  query_llm_model: z.string(),
  query_llm_base_url: z.string(),
  embedding_model: z.string(),
  reranker_model: z.string(),
  chunks_collection: z.string(),
  patents_collection: z.string(),
  retrieval_top_k_per_view: z.number(),
  patent_candidate_top_k: z.number(),
  evidence_neighbor_chunks: z.number(),
  rerank_batch_size: z.number(),
  reranker_max_context_tokens: z.number(),
  final_score_threshold: z.number(),
  weights: z.record(z.string(), z.number()),
  patent_view_url_template: z.string(),
})

export const cacheStatsSchema = z.object({
  size: z.number(),
  max_size: z.number(),
  hits: z.number(),
  misses: z.number(),
  hit_ratio: z.number(),
})

/** One NDJSON line from POST /api/search. Phase `data` is validated per phase separately. */
export const searchEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('start'),
    query: z.string(),
    cache_hit: z.boolean(),
    search_id: z.number().nullable(),
  }),
  z.object({
    type: z.literal('phase'),
    phase: z.number().int().min(1).max(7),
    name: z.string(),
    elapsed_ms: z.number(),
    data: z.unknown(),
  }),
  z.object({ type: z.literal('done'), elapsed_ms: z.number() }),
  z.object({ type: z.literal('error'), message: z.string() }),
])

export type PipelineConfig = z.infer<typeof pipelineConfigSchema>
export type CacheStats = z.infer<typeof cacheStatsSchema>
export type SearchEvent = z.infer<typeof searchEventSchema>

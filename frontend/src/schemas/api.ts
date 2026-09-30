/** Zod schemas for the FastAPI endpoints (app/api/schemas.py) and the search stream. */
import { z } from 'zod'

export const pipelineConfigSchema = z.object({
  query_llm_model: z.string(),
  query_llm_base_url: z.string(),
  embedding_model: z.string(),
  reranker_model: z.string(),
  retrieval_top_k_per_view: z.number(),
  patent_candidate_top_k: z.number(),
  evidence_neighbor_chunks: z.number(),
  rerank_batch_size: z.number(),
  reranker_max_context_tokens: z.number(),
  final_score_threshold: z.number(),
  weights: z.record(z.string(), z.number()),
  patent_view_url_template: z.string(),
})

/** A searchable Qdrant collection pair (app/models/collection.py). */
export const collectionSchema = z.object({
  name: z.string(),
  chunks_collection: z.string(),
  patents_collection: z.string(),
  chunk_count: z.number(),
  patent_count: z.number(),
})

export const collectionListSchema = z.object({
  collections: z.array(collectionSchema),
  default: z.string().nullable(),
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
    collection: z.string(),
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

/**
 * The API process's resident memory during a compare run (app/api/memory.py):
 * at the start of a step or collection, and its peak during it.
 */
export const memoryUsageSchema = z.object({
  start_bytes: z.number(),
  peak_bytes: z.number(),
  /** Only on a whole collection's run. */
  end_bytes: z.number().nullish(),
})

/** One NDJSON line from POST /api/compare. `collection` is null on the shared Phase 1 event. */
export const compareEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('start'),
    query: z.string(),
    collections: z.array(z.string()),
  }),
  z.object({
    type: z.literal('phase'),
    collection: z.string().nullable(),
    phase: z.number().int().min(1).max(7),
    name: z.string(),
    elapsed_ms: z.number(),
    data: z.unknown(),
    memory: memoryUsageSchema.nullish(),
  }),
  z.object({ type: z.literal('collection_start'), collection: z.string() }),
  z.object({
    type: z.literal('collection_done'),
    collection: z.string(),
    elapsed_ms: z.number(),
    memory: memoryUsageSchema.nullish(),
  }),
  z.object({ type: z.literal('collection_error'), collection: z.string(), message: z.string() }),
  z.object({ type: z.literal('done'), elapsed_ms: z.number() }),
  z.object({ type: z.literal('error'), message: z.string() }),
])

export type PipelineConfig = z.infer<typeof pipelineConfigSchema>
export type SearchCollection = z.infer<typeof collectionSchema>
export type CollectionList = z.infer<typeof collectionListSchema>
export type CacheStats = z.infer<typeof cacheStatsSchema>
export type SearchEvent = z.infer<typeof searchEventSchema>
export type CompareEvent = z.infer<typeof compareEventSchema>
export type MemoryUsage = z.infer<typeof memoryUsageSchema>

/** Zod schemas for saved searches (app/api/schemas.py → History*). */
import { z } from 'zod'

export const historyTopResultSchema = z.object({
  patent_id: z.string(),
  final_score: z.number(),
  title: z.string(),
})

export const historyItemSchema = z.object({
  id: z.number(),
  query: z.string(),
  use_cache: z.boolean(),
  cache_hit: z.boolean(),
  status: z.enum(['running', 'success', 'error']),
  error: z.string().nullable(),
  total_ms: z.number().nullable(),
  result_count: z.number(),
  top_results: z.array(historyTopResultSchema),
  created_at: z.string(),
})

export const historyListSchema = z.object({
  items: z.array(historyItemSchema),
  total: z.number(),
})

/** Phase data is validated per phase later (see lib/history.ts). */
export const historyDetailSchema = historyItemSchema.extend({
  phases: z.record(
    z.string(),
    z.object({ name: z.string(), elapsed_ms: z.number(), data: z.unknown() }),
  ),
})

export const historyFilterSchema = z.object({
  q: z.string().trim().max(200, 'Filter must be 200 characters or fewer.'),
})

export type HistoryItem = z.infer<typeof historyItemSchema>
export type HistoryList = z.infer<typeof historyListSchema>
export type HistoryDetail = z.infer<typeof historyDetailSchema>

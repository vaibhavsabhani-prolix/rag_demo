/** Key Feature Search validation & result schemas. */
import { z } from 'zod'

export const keyFeatureFormSchema = z.object({
  problem: z
    .string()
    .trim()
    .min(3, 'Please describe the problem you are trying to solve (at least 3 characters).'),
  invention_title: z
    .string()
    .trim()
    .min(2, 'Please enter the invention title or technology domain (at least 2 characters).'),
  invention_details: z
    .string()
    .trim()
    .min(10, 'Please provide invention details or disclosure (at least 10 characters).'),
  collection: z.string().min(1, 'Select a collection to search.'),
  score_threshold: z
    .number()
    .min(0, 'Threshold score must be at least 0.0')
    .max(1, 'Threshold score must be at most 1.0'),
})

export type KeyFeatureFormValues = z.infer<typeof keyFeatureFormSchema>

export const keyFeatureItemSchema = z.object({
  feature_id: z.number().int(),
  feature: z.string(),
})

export type KeyFeatureItem = z.infer<typeof keyFeatureItemSchema>

export const featureChunkResultSchema = z.object({
  chunk_id: z.number(),
  patent_id: z.string(),
  score: z.number(),
  text: z.string(),
  section: z.string().nullable().optional(),
  title: z.string().nullable().optional(),
  point_id: z.string().nullable().optional(),
  document_chunk_index: z.number().nullable().optional(),
  token_count: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).default({}),
})

export type FeatureChunkResult = z.infer<typeof featureChunkResultSchema>

export const featureSearchResultSchema = z.object({
  feature_id: z.number().int(),
  feature: z.string(),
  chunks: z.array(featureChunkResultSchema).default([]),
})

export type FeatureSearchResult = z.infer<typeof featureSearchResultSchema>

export const keyFeatureSearchResponseSchema = z.object({
  problem: z.string(),
  invention_title: z.string(),
  invention_details: z.string(),
  collection: z.string(),
  score_threshold: z.number().nullable().optional(),
  total_features: z.number(),
  results: z.array(featureSearchResultSchema),
  timings: z.record(z.string(), z.number()).default({}),
})

export type KeyFeatureSearchResponse = z.infer<typeof keyFeatureSearchResponseSchema>

export const keyFeatureEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('start'),
    problem: z.string().optional(),
    invention_title: z.string().optional(),
    collection: z.string().optional(),
    score_threshold: z.number().optional(),
  }),
  z.object({
    type: z.literal('phase'),
    phase: z.number(),
    name: z.string(),
    elapsed_ms: z.number(),
    data: z.unknown(),
  }),
  z.object({
    type: z.literal('done'),
    elapsed_ms: z.number(),
    data: keyFeatureSearchResponseSchema.optional(),
  }),
  z.object({
    type: z.literal('error'),
    message: z.string(),
  }),
])

export type KeyFeatureEvent = z.infer<typeof keyFeatureEventSchema>

/** Key Feature Search validation & result schemas. */
import { z } from 'zod'
import { chunkHighlightSchema } from '@/schemas/pipeline'

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
  retrieval_limit: z
    .number()
    .int()
    .min(10, 'Candidate limit must be at least 10')
    .max(10000, 'Candidate limit cannot exceed 10,000'),
  top_k: z
    .number()
    .int()
    .min(1, 'Top chunks must be at least 1')
    .max(200, 'Top chunks cannot exceed 200'),
  score_threshold: z
    .number()
    .min(0, 'Threshold score must be at least 0.0')
    .max(1, 'Threshold score must be at most 1.0'),
  rerank_score_threshold: z
    .number()
    .min(0, 'Reranking threshold score must be at least 0.0')
    .max(1, 'Reranking threshold score must be at most 1.0'),
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
  feature_id: z.number().int().nullish(),
  score: z.number(),
  retrieval_score: z.number().nullable().optional(),
  reranker_score: z.number().nullable().optional(),
  text: z.string(),
  section: z.string().nullable().optional(),
  title: z.string().nullable().optional(),
  point_id: z.string().nullable().optional(),
  document_chunk_index: z.number().nullable().optional(),
  token_count: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).default({}),
  highlight: chunkHighlightSchema.nullish(),
})

export type FeatureChunkResult = z.infer<typeof featureChunkResultSchema>

export const featurePatentResultSchema = z.object({
  patent_id: z.string(),
  title: z.string().nullable().optional(),
  score: z.number(),
  best_reranker_score: z.number().nullable().optional(),
  best_retrieval_score: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).default({}),
  chunks: z.array(featureChunkResultSchema).default([]),
})

export type FeaturePatentResult = z.infer<typeof featurePatentResultSchema>

export const stepPatentInfoSchema = z.object({
  patent_id: z.string(),
  title: z.string().nullable().optional(),
  retrieval_score: z.number().nullable().optional(),
  reranker_score: z.number().nullable().optional(),
  chunks_count: z.number().int().default(1),
  rank: z.number().int().nullable().optional(),
  stage: z.string().default('retrieval'),
  filter_reason: z.string().nullable().optional(),
  passed: z.boolean().default(true),
  chunks: z.array(featureChunkResultSchema).default([]),
})

export type StepPatentInfo = z.infer<typeof stepPatentInfoSchema>

export const featureSearchResultSchema = z.object({
  feature_id: z.number().int(),
  feature: z.string(),
  chunks: z.array(featureChunkResultSchema).default([]),
  patents: z.array(featurePatentResultSchema).default([]),
  vector_retrieval_patents: z.array(stepPatentInfoSchema).default([]),
  rerank_filtered_patents: z.array(stepPatentInfoSchema).default([]),
  topk_pruned_patents: z.array(stepPatentInfoSchema).default([]),
})

export type FeatureSearchResult = z.infer<typeof featureSearchResultSchema>

export const keyFeatureSearchResponseSchema = z.object({
  problem: z.string(),
  invention_title: z.string(),
  invention_details: z.string(),
  collection: z.string(),
  score_threshold: z.number().nullable().optional(),
  rerank_score_threshold: z.number().nullable().optional(),
  top_k: z.number().optional(),
  retrieval_limit: z.number().optional(),
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
    rerank_score_threshold: z.number().optional(),
    top_k: z.number().optional(),
    retrieval_limit: z.number().optional(),
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

export const keyFeatureCompareFormSchema = z.object({
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
  retrieval_limit: z
    .number()
    .int()
    .min(10, 'Candidate limit must be at least 10')
    .max(10000, 'Candidate limit cannot exceed 10,000'),
  top_k: z
    .number()
    .int()
    .min(1, 'Top chunks must be at least 1')
    .max(200, 'Top chunks cannot exceed 200'),
  score_threshold: z
    .number()
    .min(0, 'Threshold score must be at least 0.0')
    .max(1, 'Threshold score must be at most 1.0'),
  rerank_score_threshold: z
    .number()
    .min(0, 'Reranking threshold score must be at least 0.0')
    .max(1, 'Reranking threshold score must be at most 1.0'),
})

export type KeyFeatureCompareFields = z.infer<typeof keyFeatureCompareFormSchema>

export type KeyFeatureCompareFormValues = KeyFeatureCompareFields & {
  collections: string[]
  features?: KeyFeatureItem[]
}

export const keyFeatureCompareEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('start'),
    problem: z.string().optional(),
    invention_title: z.string().optional(),
    collections: z.array(z.string()).optional(),
    score_threshold: z.number().optional(),
    rerank_score_threshold: z.number().optional(),
    top_k: z.number().optional(),
    retrieval_limit: z.number().optional(),
  }),
  z.object({
    type: z.literal('phase'),
    phase: z.number(),
    name: z.string(),
    elapsed_ms: z.number(),
    data: z.unknown(),
  }),
  z.object({
    type: z.literal('collection_start'),
    collection: z.string(),
  }),
  z.object({
    type: z.literal('collection_phase'),
    collection: z.string(),
    phase: z.number(),
    name: z.string(),
    elapsed_ms: z.number(),
  }),
  z.object({
    type: z.literal('collection_done'),
    collection: z.string(),
    elapsed_ms: z.number(),
    data: keyFeatureSearchResponseSchema,
  }),
  z.object({
    type: z.literal('collection_error'),
    collection: z.string(),
    message: z.string(),
  }),
  z.object({
    type: z.literal('done'),
    elapsed_ms: z.number(),
  }),
  z.object({
    type: z.literal('error'),
    message: z.string(),
  }),
])

export type KeyFeatureCompareEvent = z.infer<typeof keyFeatureCompareEventSchema>

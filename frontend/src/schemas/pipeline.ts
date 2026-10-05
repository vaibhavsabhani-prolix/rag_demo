/**
 * Zod schemas for each pipeline phase result.
 * They mirror the Pydantic models in app/models/*.py and are the single
 * source of truth for the frontend types (via z.infer).
 */
import { z } from 'zod'

const metadata = z.record(z.string(), z.unknown())
const timings = z.record(z.string(), z.number())

// Phase 1 — Query Understanding (app/models/parsed_query.py)
export const relationshipSchema = z.object({
  subject: z.string(),
  relation: z.string(),
  object: z.string(),
  context: z.string().nullish(),
})

export const attributeSchema = z.object({
  concept: z.string(),
  name: z.string(),
  value: z.string(),
})

const scalar = z.union([z.string(), z.number()])

export const metadataFilterSchema = z.object({
  field: z.string(),
  operator: z.string(),
  value: z.union([scalar, z.array(scalar)]),
  raw_field: z.string().nullish(),
})

export const parsedQuerySchema = z.object({
  original_query: z.string(),
  semantic_query: z.string(),
  concepts: z.array(z.string()),
  relationships: z.array(relationshipSchema),
  attributes: z.array(attributeSchema),
  requirements: z.array(z.string()),
  constraints: z.array(z.string()),
  exclusions: z.array(z.string()),
  metadata_filters: z.array(metadataFilterSchema),
  is_metadata_only: z.boolean(),
})

// Phase 2 & 3 — Candidates (app/models/candidate.py)
export const candidateChunkSchema = z.object({
  point_id: z.string(),
  patent_id: z.string(),
  chunk_id: z.number(),
  score: z.number(),
  matched_views: z.array(z.string()),
  section: z.string().nullish(),
  text: z.string().nullish(),
  document_chunk_index: z.number().nullish(),
  token_count: z.number().nullish(),
})

export const candidatePatentSchema = z.object({
  patent_id: z.string(),
  best_chunk_id: z.number().nullish(),
  retrieval_score: z.number(),
  matched_views: z.array(z.string()),
  chunk_count: z.number(),
  chunks: z.array(candidateChunkSchema),
  metadata,
})

export const candidateRetrievalSchema = z.object({
  candidates: z.array(candidatePatentSchema),
  retrieval_views: z.record(z.string(), z.string()),
  total_chunk_hits: z.number(),
  unique_patents: z.number(),
  is_metadata_only: z.boolean(),
  timings,
})

export const filterDiagnosticSchema = z.object({
  field: z.string(),
  operator: z.string(),
  value: z.unknown(),
  passed: z.number(),
  failed: z.number(),
})

export const filteredCandidatesSchema = z.object({
  candidates: z.array(candidatePatentSchema),
  total_before: z.number(),
  total_after: z.number(),
  filtered_count: z.number(),
  diagnostics: z.array(filterDiagnosticSchema),
  is_metadata_only: z.boolean(),
  filter_time_ms: z.number(),
})

// Evidence chunk shape (app/models/evidence.py) — no longer its own phase;
// Phase 4/5 build evidence straight from Phase 2/3's candidate chunks.
export const evidenceChunkSchema = z.object({
  patent_id: z.string(),
  chunk_id: z.number(),
  text: z.string(),
  retrieval_score: z.number(),
  retrieval_source: z.string(),
  section: z.string().nullish(),
  document_chunk_index: z.number().nullish(),
  token_count: z.number().nullish(),
})

// Phase 4 — Verification (app/models/verification.py)
export const relationshipVerificationSchema = z.object({
  relationship_index: z.number(),
  subject: z.string(),
  relation: z.string(),
  object: z.string(),
  supported: z.boolean(),
  status: z.string(),
  confidence: z.number(),
  evidence_chunk_ids: z.array(z.number()),
  explanation: z.string().nullish(),
})

export const requirementVerificationSchema = z.object({
  requirement_index: z.number(),
  requirement: z.string(),
  supported: z.boolean(),
  confidence: z.number(),
  evidence_chunk_ids: z.array(z.number()),
  explanation: z.string().nullish(),
})

const verificationCounts = {
  supported_count: z.number(),
  unsupported_count: z.number(),
  contradicted_count: z.number(),
  unknown_count: z.number(),
}

export const patentVerificationSchema = z.object({
  patent_id: z.string(),
  relationships: z.array(relationshipVerificationSchema),
  requirements: z.array(requirementVerificationSchema),
  relationship_coverage: z.number(),
  requirement_coverage: z.number(),
  ...verificationCounts,
  metadata,
  candidate_score: z.number(),
})

export const verificationBatchSchema = z.object({
  verified_patents: z.array(patentVerificationSchema),
  total_evaluated: z.number(),
  fully_supported_count: z.number(),
  partially_supported_count: z.number(),
  unsupported_count: z.number(),
  timings,
})

// Phase 5 — Reranking (app/models/reranking.py)
/** Character spans of a chunk's text that match the query. */
export const chunkHighlightSchema = z.object({
  /** Sentences the cross-encoder scored as matching the query in meaning. */
  sentences: z.array(z.object({ start: z.number(), end: z.number(), score: z.number(), strong: z.boolean() })),
  /** Occurrences of the query's own words and concepts. */
  terms: z.array(z.object({ start: z.number(), end: z.number() })),
})

export const rerankedChunkSchema = evidenceChunkSchema.extend({
  reranker_score: z.number(),
  // Missing on searches saved before highlighting was added.
  highlight: chunkHighlightSchema.nullish(),
})

export const rerankedPatentSchema = z.object({
  patent_id: z.string(),
  metadata,
  candidate_score: z.number(),
  relationship_coverage: z.number(),
  requirement_coverage: z.number(),
  relationships: z.array(relationshipVerificationSchema),
  requirements: z.array(requirementVerificationSchema),
  ...verificationCounts,
  evidence: z.array(rerankedChunkSchema),
  best_reranker_score: z.number(),
  avg_reranker_score: z.number(),
})

export const rerankBatchSchema = z.object({
  reranked_patents: z.array(rerankedPatentSchema),
  total_candidates: z.number(),
  total_chunks_reranked: z.number(),
  total_sentences_scored: z.number().default(0),
  total_requests: z.number(),
  truncated_chunks_count: z.number(),
  reranking_query: z.string(),
  timings,
})

// Phase 6 — Final scoring (app/models/scoring.py)
export const scoreBreakdownSchema = z.object({
  relationship_score: z.number(),
  requirement_score: z.number(),
  reranker_score: z.number(),
  retrieval_score: z.number(),
  relationship_weighted: z.number(),
  requirement_weighted: z.number(),
  reranker_weighted: z.number(),
  retrieval_weighted: z.number(),
  raw_total: z.number(),
})

export const finalPatentSchema = z.object({
  patent_id: z.string(),
  // Both null for a metadata-only query: nothing semantic to score or rank by.
  final_score: z.number().nullable(),
  score_breakdown: scoreBreakdownSchema.nullable(),
  metadata,
  relationship_coverage: z.number(),
  requirement_coverage: z.number(),
  best_reranker_score: z.number(),
  candidate_score: z.number(),
  ...verificationCounts,
  relationships: z.array(relationshipVerificationSchema),
  requirements: z.array(requirementVerificationSchema),
  evidence: z.array(rerankedChunkSchema),
})

export const finalSearchSchema = z.object({
  results: z.array(finalPatentSchema),
  total_candidates_evaluated: z.number(),
  passed_threshold_count: z.number(),
  rejected_count: z.number(),
  threshold_used: z.number(),
  weights_used: z.record(z.string(), z.number()),
  timings,
})

/** Phase number -> schema of the result that phase produces. */
export const phaseSchemas = {
  1: parsedQuerySchema,
  2: candidateRetrievalSchema,
  3: filteredCandidatesSchema,
  4: verificationBatchSchema,
  5: rerankBatchSchema,
  6: finalSearchSchema,
} as const

export type PhaseNumber = keyof typeof phaseSchemas
export type PhaseResults = { [K in PhaseNumber]: z.infer<(typeof phaseSchemas)[K]> }

export const PHASES: { id: PhaseNumber; title: string; short: string }[] = [
  { id: 1, title: 'Query Understanding', short: 'Understand' },
  { id: 2, title: 'Candidate Retrieval', short: 'Retrieve' },
  { id: 3, title: 'Metadata Filtering', short: 'Filter' },
  { id: 4, title: 'Relationship Verification', short: 'Verify' },
  { id: 5, title: 'Cross-Encoder Reranking', short: 'Rerank' },
  { id: 6, title: 'Final Scoring', short: 'Score' },
]

export type ParsedQuery = z.infer<typeof parsedQuerySchema>
export type CandidatePatent = z.infer<typeof candidatePatentSchema>
export type EvidenceChunk = z.infer<typeof evidenceChunkSchema>
export type RelationshipVerification = z.infer<typeof relationshipVerificationSchema>
export type RequirementVerification = z.infer<typeof requirementVerificationSchema>
export type RerankedChunk = z.infer<typeof rerankedChunkSchema>
export type ChunkHighlight = z.infer<typeof chunkHighlightSchema>
export type ScoreBreakdown = z.infer<typeof scoreBreakdownSchema>
export type FinalPatent = z.infer<typeof finalPatentSchema>

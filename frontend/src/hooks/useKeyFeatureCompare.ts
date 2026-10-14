import { useState, useRef, useCallback } from 'react'
import { api, ApiError } from '@/lib/api'
import type {
  KeyFeatureCompareFormValues,
  KeyFeatureItem,
  KeyFeatureSearchResponse,
} from '@/schemas/keyfeature'

export interface CollectionRunState {
  collection: string
  status: 'idle' | 'running' | 'success' | 'error'
  currentStep?: number
  stepName?: string
  response?: KeyFeatureSearchResponse
  elapsedMs?: number
  error?: string
}

export interface KeyFeatureCompareState {
  status: 'idle' | 'running' | 'success' | 'error'
  currentStep: number | null
  stepName: string | null
  extractedFeatures: KeyFeatureItem[]
  collections: string[]
  currentCollection: string | null
  runs: Record<string, CollectionRunState>
  elapsedMs: number | null
  error: string | null
  inputs: KeyFeatureCompareFormValues | null
}

const INITIAL_STATE: KeyFeatureCompareState = {
  status: 'idle',
  currentStep: null,
  stepName: null,
  extractedFeatures: [],
  collections: [],
  currentCollection: null,
  runs: {},
  elapsedMs: null,
  error: null,
  inputs: null,
}

export function useKeyFeatureCompare() {
  const [state, setState] = useState<KeyFeatureCompareState>(INITIAL_STATE)
  const abortRef = useRef<AbortController | null>(null)

  const start = useCallback(async (values: KeyFeatureCompareFormValues) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const initialRuns: Record<string, CollectionRunState> = {}
    for (const c of values.collections) {
      initialRuns[c] = {
        collection: c,
        status: 'idle',
      }
    }

    const hasPreExtractedFeatures = Boolean(values.features && values.features.length > 0)

    setState({
      ...INITIAL_STATE,
      status: 'running',
      currentStep: hasPreExtractedFeatures ? 2 : 1,
      stepName: hasPreExtractedFeatures
        ? 'Reusing extracted key features (LLM skipped)…'
        : 'Extracting key technical features with LLM…',
      extractedFeatures: values.features || [],
      collections: values.collections,
      runs: initialRuns,
      inputs: values,
    })

    try {
      await api.streamKeyFeatureCompare(
        {
          problem: values.problem,
          invention_title: values.invention_title,
          invention_details: values.invention_details,
          features: values.features,
          collections: values.collections,
          score_threshold: values.score_threshold,
          rerank_score_threshold: values.rerank_score_threshold,
          top_k: values.top_k,
          retrieval_limit: values.retrieval_limit,
        },
        (event) => {
          switch (event.type) {
            case 'start':
              setState((prev) => ({
                ...prev,
                currentStep: hasPreExtractedFeatures ? 2 : 1,
                stepName: hasPreExtractedFeatures
                  ? 'Reusing extracted key features (LLM skipped)…'
                  : 'Extracting key technical features with LLM…',
              }))
              break

            case 'phase': {
              const phaseNum = event.phase
              const data = event.data as Record<string, unknown>

              if (phaseNum === 1) {
                const rawFeatures = (data?.key_features as KeyFeatureItem[]) || []
                setState((prev) => ({
                  ...prev,
                  currentStep: 2,
                  stepName: 'Generating independent vector embeddings for features…',
                  extractedFeatures: rawFeatures.length > 0 ? rawFeatures : prev.extractedFeatures,
                }))
              } else if (phaseNum === 2) {
                setState((prev) => ({
                  ...prev,
                  currentStep: 3,
                  stepName: 'Comparing collections…',
                }))
              }
              break
            }

            case 'collection_start': {
              setState((prev) => ({
                ...prev,
                currentCollection: event.collection,
                stepName: `Searching collection “${event.collection}”…`,
                runs: {
                  ...prev.runs,
                  [event.collection]: {
                    collection: event.collection,
                    status: 'running',
                    currentStep: 3,
                    stepName: 'Retrieving candidates…',
                  },
                },
              }))
              break
            }

            case 'collection_phase': {
              setState((prev) => {
                const existing = prev.runs[event.collection] || {
                  collection: event.collection,
                  status: 'running',
                }
                return {
                  ...prev,
                  runs: {
                    ...prev.runs,
                    [event.collection]: {
                      ...existing,
                      status: 'running',
                      currentStep: event.phase,
                      stepName: event.name,
                    },
                  },
                }
              })
              break
            }

            case 'collection_done': {
              setState((prev) => ({
                ...prev,
                runs: {
                  ...prev.runs,
                  [event.collection]: {
                    collection: event.collection,
                    status: 'success',
                    elapsedMs: event.elapsed_ms,
                    response: event.data,
                  },
                },
              }))
              break
            }

            case 'collection_error': {
              setState((prev) => ({
                ...prev,
                runs: {
                  ...prev.runs,
                  [event.collection]: {
                    collection: event.collection,
                    status: 'error',
                    error: event.message,
                  },
                },
              }))
              break
            }

            case 'done': {
              setState((prev) => ({
                ...prev,
                status: 'success',
                currentStep: null,
                currentCollection: null,
                stepName: null,
                elapsedMs: event.elapsed_ms,
              }))
              break
            }

            case 'error':
              throw new ApiError(event.message)
          }
        },
        controller.signal,
      )
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return
      const message = err instanceof Error ? err.message : String(err)
      setState((prev) => ({
        ...prev,
        status: 'error',
        currentStep: null,
        currentCollection: null,
        stepName: null,
        error: message,
      }))
    }
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setState((prev) => {
      const hasAnySuccess = Object.values(prev.runs).some((r) => r.status === 'success')
      return {
        ...prev,
        status: hasAnySuccess ? 'success' : 'idle',
        currentStep: null,
        currentCollection: null,
        stepName: null,
      }
    })
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState(INITIAL_STATE)
  }, [])

  const clearFeatures = useCallback(() => {
    setState((prev) => ({
      ...prev,
      extractedFeatures: [],
    }))
  }, [])

  return {
    ...state,
    start,
    stop,
    reset,
    clearFeatures,
    isPending: state.status === 'running',
  }
}


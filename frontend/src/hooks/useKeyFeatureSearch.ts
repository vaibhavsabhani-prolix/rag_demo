import { useState, useRef, useCallback } from 'react'
import { api, ApiError } from '@/lib/api'
import type {
  KeyFeatureFormValues,
  KeyFeatureItem,
  FeatureSearchResult,
  KeyFeatureSearchResponse,
} from '@/schemas/keyfeature'

export interface KeyFeatureSearchState {
  status: 'idle' | 'running' | 'success' | 'error'
  currentStep: number | null
  stepName: string | null
  extractedFeatures: KeyFeatureItem[]
  results: FeatureSearchResult[]
  response: KeyFeatureSearchResponse | null
  error: string | null
  elapsedMs: number | null
  inputs: KeyFeatureFormValues | null
}

const INITIAL_STATE: KeyFeatureSearchState = {
  status: 'idle',
  currentStep: null,
  stepName: null,
  extractedFeatures: [],
  results: [],
  response: null,
  error: null,
  elapsedMs: null,
  inputs: null,
}

export function useKeyFeatureSearch() {
  const [state, setState] = useState<KeyFeatureSearchState>(INITIAL_STATE)
  const abortRef = useRef<AbortController | null>(null)

  const search = useCallback(async (values: KeyFeatureFormValues) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setState({
      ...INITIAL_STATE,
      status: 'running',
      currentStep: 1,
      stepName: 'Extracting 5–15 key technical features from disclosure…',
      inputs: values,
    })

    try {
      await api.streamKeyFeatureSearch(
        {
          problem: values.problem,
          invention_title: values.invention_title,
          invention_details: values.invention_details,
          collection: values.collection,
          top_k: values.top_k,
        },
        (event) => {
          switch (event.type) {
            case 'start':
              setState((prev) => ({
                ...prev,
                currentStep: 1,
                stepName: 'Extracting 5–15 key technical features from disclosure…',
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
                  stepName: 'Generating independent embeddings for each key feature…',
                  extractedFeatures: rawFeatures,
                }))
              } else if (phaseNum === 2) {
                setState((prev) => ({
                  ...prev,
                  currentStep: 3,
                  stepName: 'Searching Qdrant independently for each key feature…',
                }))
              } else if (phaseNum === 3) {
                const rawResults = (data?.results as FeatureSearchResult[]) || []
                setState((prev) => ({
                  ...prev,
                  results: rawResults,
                }))
              }
              break
            }

            case 'done': {
              const responseData = event.data as KeyFeatureSearchResponse
              setState((prev) => ({
                ...prev,
                status: 'success',
                currentStep: null,
                stepName: null,
                elapsedMs: event.elapsed_ms,
                response: responseData || null,
                results: responseData?.results || prev.results,
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
        stepName: null,
        error: message,
      }))
    }
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState(INITIAL_STATE)
  }, [])

  return {
    ...state,
    search,
    reset,
    isPending: state.status === 'running',
  }
}


/**
 * Runs one query against several collections as a React Query mutation and
 * streams every phase result into the compare slice as it arrives.
 */
import { useMutation } from '@tanstack/react-query'
import { useRef } from 'react'
import { z } from 'zod'
import { api, ApiError } from '@/lib/api'
import { phaseSchemas, type PhaseNumber } from '@/schemas/pipeline'
import { useAppDispatch } from '@/store'
import {
  collectionFailed,
  collectionStarted,
  collectionSucceeded,
  compareFailed,
  comparePhaseReceived,
  compareStarted,
  compareStreamOpened,
  compareSucceeded,
  type ComparePhasePayload,
} from '@/store/compareSlice'

export interface CompareValues {
  query: string
  collections: string[]
  useCache: boolean
}

export function useCompare() {
  const dispatch = useAppDispatch()
  const abortRef = useRef<AbortController | null>(null)

  const mutation = useMutation({
    mutationFn: async ({ query, collections, useCache }: CompareValues) => {
      // Starting a new comparison cancels the one still streaming.
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      dispatch(compareStarted({ query, collections }))
      let finished = false

      try {
        await api.streamCompare(
          { query, collections, use_cache: useCache },
          (event) => {
            switch (event.type) {
              case 'start':
                dispatch(compareStreamOpened({ cacheHit: event.cache_hit }))
                break
              case 'phase': {
                const phase = event.phase as PhaseNumber
                const data = phaseSchemas[phase].parse(event.data)
                dispatch(
                  comparePhaseReceived({
                    collection: event.collection,
                    phase,
                    name: event.name,
                    elapsedMs: event.elapsed_ms,
                    data,
                  } as ComparePhasePayload),
                )
                break
              }
              case 'collection_start':
                dispatch(collectionStarted(event.collection))
                break
              case 'collection_done':
                dispatch(collectionSucceeded({ collection: event.collection, elapsedMs: event.elapsed_ms }))
                break
              case 'collection_error':
                dispatch(collectionFailed({ collection: event.collection, message: event.message }))
                break
              case 'done':
                finished = true
                dispatch(compareSucceeded(event.elapsed_ms))
                break
              case 'error':
                throw new ApiError(event.message)
            }
          },
          controller.signal,
        )
      } catch (error) {
        if (!controller.signal.aborted) throw error
        // Replaced by a newer comparison, which owns the state now.
        if (abortRef.current !== controller) return
        throw new ApiError('Stopped.')
      }

      if (!finished) throw new ApiError('The comparison stream ended unexpectedly.')
    },
    onError: (error) => {
      const message =
        error instanceof z.ZodError ? `Unexpected data from the server:\n${z.prettifyError(error)}` : error.message
      dispatch(compareFailed(message))
    },
  })

  /** Stop the running comparison; the server skips the collections not started yet. */
  const stop = () => abortRef.current?.abort()

  return { ...mutation, stop }
}

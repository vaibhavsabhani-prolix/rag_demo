/**
 * Runs a search as a React Query mutation and streams each phase result
 * into the Redux store as soon as the backend reports it.
 */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useRef } from 'react'
import { z } from 'zod'
import { api, ApiError } from '@/lib/api'
import { phaseSchemas, type PhaseNumber } from '@/schemas/pipeline'
import type { SearchFormValues } from '@/schemas/search'
import { useAppDispatch } from '@/store'
import {
  phaseReceived,
  searchFailed,
  searchStarted,
  searchSucceeded,
  streamOpened,
  type PhaseReceivedPayload,
} from '@/store/searchSlice'
import { queryKeys } from './queries'

export function useSearch() {
  const dispatch = useAppDispatch()
  const queryClient = useQueryClient()
  const abortRef = useRef<AbortController | null>(null)

  return useMutation({
    mutationFn: async ({ query, useCache }: SearchFormValues) => {
      // Starting a new search cancels the one still streaming.
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      dispatch(searchStarted(query))
      let finished = false

      await api.streamSearch(
        { query, use_cache: useCache },
        (event) => {
          switch (event.type) {
            case 'start':
              dispatch(streamOpened({ cacheHit: event.cache_hit, searchId: event.search_id }))
              break
            case 'phase': {
              const phase = event.phase as PhaseNumber
              const data = phaseSchemas[phase].parse(event.data)
              dispatch(
                phaseReceived({
                  phase,
                  name: event.name,
                  elapsedMs: event.elapsed_ms,
                  data,
                } as PhaseReceivedPayload),
              )
              break
            }
            case 'done':
              finished = true
              dispatch(searchSucceeded(event.elapsed_ms))
              break
            case 'error':
              throw new ApiError(event.message)
          }
        },
        controller.signal,
      )

      if (!finished) throw new ApiError('The search stream ended unexpectedly.')
    },
    onError: (error) => {
      if (error.name === 'AbortError') return
      const message =
        error instanceof z.ZodError
          ? `Unexpected data from the server:\n${z.prettifyError(error)}`
          : error.message
      dispatch(searchFailed(message))
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cacheStats })
      queryClient.invalidateQueries({ queryKey: queryKeys.history })
    },
  })
}

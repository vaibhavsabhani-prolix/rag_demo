/**
 * Runs one query against several collections as a React Query mutation and
 * streams every phase result into the compare slice as it arrives.
 *
 * Call it once per page and share the result: `stop` only aborts requests
 * started through the same instance.
 */
import { useMutation } from '@tanstack/react-query'
import { useRef } from 'react'
import { z } from 'zod'
import { api, ApiError } from '@/lib/api'
import { phaseSchemas, type PhaseNumber } from '@/schemas/pipeline'
import { useAppDispatch, useAppSelector } from '@/store'
import {
  collectionFailed,
  collectionStarted,
  collectionSucceeded,
  compareFailed,
  comparePhaseReceived,
  compareStarted,
  compareSucceeded,
  runsRequested,
  type ComparePhasePayload,
} from '@/store/compareSlice'
import { selectCompare } from '@/store/selectors'

export interface CompareValues {
  query: string
  collections: string[]
  /** Run only the first collection now; the others wait to be run one at a time. */
  oneByOne?: boolean
}

/** What one request streams: some collections of the current comparison. */
interface RunValues {
  query: string
  collections: string[]
}

export function useCompare() {
  const dispatch = useAppDispatch()
  const { query } = useAppSelector(selectCompare)
  const abortRef = useRef<AbortController | null>(null)

  const mutation = useMutation({
    mutationFn: async ({ query, collections }: RunValues) => {
      // Starting a new request cancels the one still streaming.
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      dispatch(runsRequested(collections))
      let finished = false

      try {
        await api.streamCompare(
          { query, collections },
          (event) => {
            switch (event.type) {
              case 'start':
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
                    memory: event.memory ?? undefined,
                  } as ComparePhasePayload),
                )
                break
              }
              case 'collection_start':
                dispatch(collectionStarted(event.collection))
                break
              case 'collection_done':
                dispatch(
                  collectionSucceeded({
                    collection: event.collection,
                    elapsedMs: event.elapsed_ms,
                    memory: event.memory ?? undefined,
                  }),
                )
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
        // Replaced by a newer request, which owns the state now.
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

  /** Start a new comparison: every collection in a row, or just the first when `oneByOne`. */
  const start = ({ query, collections, oneByOne = false }: CompareValues) => {
    dispatch(compareStarted({ query, collections }))
    mutation.mutate({ query, collections: oneByOne ? collections.slice(0, 1) : collections })
  }

  /** Run (or rerun) some collections of the current comparison, keeping the others' results. */
  const run = (collections: string[]) => mutation.mutate({ query, collections })

  /** Stop the running request; the server skips the collections not started yet. */
  const stop = () => abortRef.current?.abort()

  return { isPending: mutation.isPending, start, run, stop }
}

export type CompareController = ReturnType<typeof useCompare>

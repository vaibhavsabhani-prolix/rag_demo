import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { MemoryUsage } from '@/schemas/api'
import type { PhaseNumber, PhaseResults } from '@/schemas/pipeline'

export type SearchStatus = 'idle' | 'running' | 'success' | 'error'

export interface PhaseEntry<T> {
  name: string
  elapsedMs: number
  data: T
  /** Measured on compare runs only. */
  memory?: MemoryUsage
}

export type PhaseMap = { [K in PhaseNumber]?: PhaseEntry<PhaseResults[K]> }

export type PhaseReceivedPayload = {
  [K in PhaseNumber]: { phase: K } & PhaseEntry<PhaseResults[K]>
}[PhaseNumber]

/** One pipeline run — either live (this slice) or loaded from history. */
export interface SearchRun {
  query: string
  /** Name of the searched collection; null for history saved before collections were selectable. */
  collection: string | null
  status: SearchStatus
  cacheHit: boolean
  phases: PhaseMap
  totalMs?: number
  /** Memory over Phases 2–7; measured on compare runs only. */
  memory?: MemoryUsage
  error?: string
  /** History record id, once the server has saved the search. */
  searchId?: number | null
}

const initialState: SearchRun = {
  query: '',
  collection: null,
  status: 'idle',
  cacheHit: false,
  phases: {},
}

const searchSlice = createSlice({
  name: 'search',
  initialState,
  reducers: {
    searchStarted(_, action: PayloadAction<{ query: string; collection: string }>) {
      return { ...initialState, ...action.payload, status: 'running' }
    },
    streamOpened(state, action: PayloadAction<{ cacheHit: boolean; searchId: number | null }>) {
      state.cacheHit = action.payload.cacheHit
      state.searchId = action.payload.searchId
    },
    phaseReceived(state, action: PayloadAction<PhaseReceivedPayload>) {
      const { phase, ...entry } = action.payload
      ;(state.phases as Record<number, PhaseEntry<unknown>>)[phase] = entry
    },
    searchSucceeded(state, action: PayloadAction<number>) {
      state.status = 'success'
      state.totalMs = action.payload
    },
    searchFailed(state, action: PayloadAction<string>) {
      state.status = 'error'
      state.error = action.payload
    },
  },
})

export const { searchStarted, streamOpened, phaseReceived, searchSucceeded, searchFailed } = searchSlice.actions
export default searchSlice.reducer

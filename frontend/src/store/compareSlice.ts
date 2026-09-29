/**
 * One query run against several collections (POST /api/compare). Each collection
 * gets its own SearchRun, so the regular results and pipeline views can show it.
 */
import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { PhaseEntry, PhaseReceivedPayload, SearchRun, SearchStatus } from './searchSlice'

export interface CompareState {
  query: string
  status: SearchStatus
  cacheHit: boolean
  /** Collection names, in the order they run. */
  collections: string[]
  /**
   * Per collection. A run waiting its turn has status 'idle'. Its totalMs covers
   * Phases 2–7 only; Phase 1 is shared and runs once.
   */
  runs: Record<string, SearchRun>
  totalMs?: number
  error?: string
}

const initialState: CompareState = {
  query: '',
  status: 'idle',
  cacheHit: false,
  collections: [],
  runs: {},
}

export type ComparePhasePayload = PhaseReceivedPayload & { collection: string | null }

const compareSlice = createSlice({
  name: 'compare',
  initialState,
  reducers: {
    compareStarted(_, action: PayloadAction<{ query: string; collections: string[] }>) {
      const { query, collections } = action.payload
      const runs: Record<string, SearchRun> = {}
      for (const collection of collections) {
        runs[collection] = { query, collection, status: 'idle', cacheHit: false, phases: {} }
      }
      return { ...initialState, query, collections, runs, status: 'running' }
    },
    compareStreamOpened(state, action: PayloadAction<{ cacheHit: boolean }>) {
      state.cacheHit = action.payload.cacheHit
      for (const run of Object.values(state.runs)) run.cacheHit = action.payload.cacheHit
    },
    comparePhaseReceived(state, action: PayloadAction<ComparePhasePayload>) {
      const { collection, phase, ...entry } = action.payload
      // The shared Phase 1 result (collection null) belongs to every run.
      const targets = collection === null ? Object.values(state.runs) : [state.runs[collection]]
      for (const run of targets) {
        if (run) (run.phases as Record<number, PhaseEntry<unknown>>)[phase] = entry
      }
    },
    collectionStarted(state, action: PayloadAction<string>) {
      const run = state.runs[action.payload]
      if (run) run.status = 'running'
    },
    collectionSucceeded(state, action: PayloadAction<{ collection: string; elapsedMs: number }>) {
      const run = state.runs[action.payload.collection]
      if (run) {
        run.status = 'success'
        run.totalMs = action.payload.elapsedMs
      }
    },
    collectionFailed(state, action: PayloadAction<{ collection: string; message: string }>) {
      const run = state.runs[action.payload.collection]
      if (run) {
        run.status = 'error'
        run.error = action.payload.message
      }
    },
    compareSucceeded(state, action: PayloadAction<number>) {
      state.status = 'success'
      state.totalMs = action.payload
    },
    compareFailed(state, action: PayloadAction<string>) {
      state.status = 'error'
      state.error = action.payload
      // Runs still going or waiting never finish now.
      for (const run of Object.values(state.runs)) {
        if (run.status === 'running') {
          run.status = 'error'
          run.error = action.payload
        } else if (run.status === 'idle') {
          run.status = 'error'
          run.error = 'Not run: the comparison stopped first.'
        }
      }
    },
  },
})

export const {
  compareStarted,
  compareStreamOpened,
  comparePhaseReceived,
  collectionStarted,
  collectionSucceeded,
  collectionFailed,
  compareSucceeded,
  compareFailed,
} = compareSlice.actions
export default compareSlice.reducer

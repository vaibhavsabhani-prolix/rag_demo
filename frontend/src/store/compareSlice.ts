/**
 * One query run against several collections (POST /api/compare). Each collection
 * gets its own SearchRun, so the regular results and pipeline views can show it.
 *
 * Collections can run all in a row or one by one: every request runs some of
 * the comparison's collections (`active`) and leaves the others' runs as they are.
 */
import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { MemoryUsage } from '@/schemas/api'
import type { PhaseEntry, PhaseReceivedPayload, SearchRun, SearchStatus } from './searchSlice'

export interface CompareState {
  query: string
  useCache: boolean
  /** Of the latest request. */
  status: SearchStatus
  cacheHit: boolean
  /** Collection names, in the order they run. */
  collections: string[]
  /** Collections the latest request runs. */
  active: string[]
  /** The collection run on its own most recently, shown first in the details. */
  focus?: string
  /**
   * Per collection. A run waiting its turn has status 'idle'. Its totalMs covers
   * Phases 2–7 only; Phase 1 is shared and runs once per request.
   */
  runs: Record<string, SearchRun>
  /** Set when one request ran every collection. */
  totalMs?: number
  error?: string
}

const initialState: CompareState = {
  query: '',
  useCache: true,
  status: 'idle',
  cacheHit: false,
  collections: [],
  active: [],
  runs: {},
}

export type ComparePhasePayload = PhaseReceivedPayload & { collection: string | null }

const newRun = (query: string, collection: string): SearchRun => ({
  query,
  collection,
  status: 'idle',
  cacheHit: false,
  phases: {},
})

const compareSlice = createSlice({
  name: 'compare',
  initialState,
  reducers: {
    /** A new comparison: every collection waits until a request runs it. */
    compareStarted(_, action: PayloadAction<{ query: string; collections: string[]; useCache: boolean }>) {
      const { query, collections, useCache } = action.payload
      const runs = Object.fromEntries(collections.map((c) => [c, newRun(query, c)]))
      return { ...initialState, query, useCache, collections, runs }
    },
    /** A request for these collections is starting; earlier results for them are cleared. */
    runsRequested(state, action: PayloadAction<string[]>) {
      const active = action.payload
      state.active = active
      state.status = 'running'
      state.error = undefined
      state.totalMs = undefined
      if (active.length === 1) state.focus = active[0]
      for (const name of active) state.runs[name] = newRun(state.query, name)
    },
    compareStreamOpened(state, action: PayloadAction<{ cacheHit: boolean }>) {
      state.cacheHit = action.payload.cacheHit
      for (const name of state.active) state.runs[name].cacheHit = action.payload.cacheHit
    },
    comparePhaseReceived(state, action: PayloadAction<ComparePhasePayload>) {
      const { collection, phase, ...entry } = action.payload
      // The shared Phase 1 result (collection null) belongs to every run in the request.
      const targets = collection === null ? state.active.map((name) => state.runs[name]) : [state.runs[collection]]
      for (const run of targets) {
        if (run) (run.phases as Record<number, PhaseEntry<unknown>>)[phase] = entry
      }
    },
    collectionStarted(state, action: PayloadAction<string>) {
      const run = state.runs[action.payload]
      if (run) run.status = 'running'
    },
    collectionSucceeded(
      state,
      action: PayloadAction<{ collection: string; elapsedMs: number; memory?: MemoryUsage }>,
    ) {
      const run = state.runs[action.payload.collection]
      if (run) {
        run.status = 'success'
        run.totalMs = action.payload.elapsedMs
        run.memory = action.payload.memory
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
      if (state.active.length === state.collections.length) state.totalMs = action.payload
    },
    compareFailed(state, action: PayloadAction<string>) {
      state.status = 'error'
      state.error = action.payload
      // Runs of this request still going or waiting never finish now.
      for (const name of state.active) {
        const run = state.runs[name]
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
  runsRequested,
  compareStreamOpened,
  comparePhaseReceived,
  collectionStarted,
  collectionSucceeded,
  collectionFailed,
  compareSucceeded,
  compareFailed,
} = compareSlice.actions
export default compareSlice.reducer

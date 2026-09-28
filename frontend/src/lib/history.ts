/** Converts a saved search from the API into the same shape as a live run. */
import { phaseSchemas, PHASES } from '@/schemas/pipeline'
import type { HistoryDetail } from '@/schemas/history'
import type { PhaseEntry, SearchRun } from '@/store/searchSlice'

export function historyToRun(detail: HistoryDetail): SearchRun {
  const phases: Record<number, PhaseEntry<unknown>> = {}
  for (const { id } of PHASES) {
    const saved = detail.phases[String(id)]
    if (saved) {
      phases[id] = { name: saved.name, elapsedMs: saved.elapsed_ms, data: phaseSchemas[id].parse(saved.data) }
    }
  }

  // A search still marked "running" was interrupted (e.g. the server restarted).
  const interrupted = detail.status === 'running'
  return {
    query: detail.query,
    status: interrupted ? 'error' : detail.status,
    error: interrupted ? 'This search did not finish.' : (detail.error ?? undefined),
    cacheHit: detail.cache_hit,
    totalMs: detail.total_ms ?? undefined,
    searchId: detail.id,
    phases,
  }
}

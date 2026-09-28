/** Thin API client for the FastAPI backend. Every response is validated with Zod. */
import type { z } from 'zod'
import {
  cacheStatsSchema,
  pipelineConfigSchema,
  searchEventSchema,
  type SearchEvent,
} from '@/schemas/api'
import { historyDetailSchema, historyListSchema } from '@/schemas/history'
import { uiSettingsSchema, type UiSettings } from '@/schemas/settings'

const API_BASE = '/api'

export class ApiError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.status = status
  }
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    // FastAPI validation errors come back as { detail: [{ msg }] }
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg: string }) => d.msg).join(', ')
    if (typeof body?.detail === 'string') return body.detail
  } catch {
    // fall through to the status text
  }
  return `${res.status} ${res.statusText}`
}

async function request<T>(path: string, schema: z.ZodType<T>, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) throw new ApiError(await readError(res), res.status)
  return schema.parse(await res.json())
}

/** For endpoints that return no body (204). */
async function send(path: string, init: RequestInit): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) throw new ApiError(await readError(res), res.status)
}

export interface HistoryParams {
  limit: number
  offset: number
  q?: string
}

export const api = {
  getConfig: () => request('/config', pipelineConfigSchema),
  getCacheStats: () => request('/cache', cacheStatsSchema),
  clearCache: () => request('/cache', cacheStatsSchema, { method: 'DELETE' }),

  getHistory: ({ limit, offset, q }: HistoryParams) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    if (q) params.set('q', q)
    return request(`/history?${params}`, historyListSchema)
  },
  getHistoryItem: (id: number) => request(`/history/${id}`, historyDetailSchema),
  deleteHistoryItem: (id: number) => send(`/history/${id}`, { method: 'DELETE' }),
  clearHistory: () => send('/history', { method: 'DELETE' }),

  getUiSettings: () => request('/settings/ui', uiSettingsSchema),
  saveUiSettings: (settings: UiSettings) =>
    request('/settings/ui', uiSettingsSchema, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    }),

  /**
   * Run a search and call `onEvent` for every NDJSON event as it arrives.
   * Resolves once the stream closes.
   */
  async streamSearch(
    body: { query: string; use_cache: boolean },
    onEvent: (event: SearchEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const res = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    if (!res.ok || !res.body) throw new ApiError(await readError(res), res.status)

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
    let buffer = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += value
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line.trim()) onEvent(searchEventSchema.parse(JSON.parse(line)))
      }
    }
    if (buffer.trim()) onEvent(searchEventSchema.parse(JSON.parse(buffer)))
  },
}

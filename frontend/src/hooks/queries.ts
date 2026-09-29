/** React Query hooks for the backend's read endpoints and mutations. */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type HistoryParams } from '@/lib/api'
import { historyToRun } from '@/lib/history'
import { DEFAULT_UI_SETTINGS } from '@/schemas/settings'

export const queryKeys = {
  config: ['config'] as const,
  collections: ['collections'] as const,
  cacheStats: ['cache-stats'] as const,
  history: ['history'] as const,
  historyList: (params: HistoryParams) => ['history', 'list', params] as const,
  historyDetail: (id: number) => ['history', 'detail', id] as const,
  uiSettings: ['ui-settings'] as const,
}

export function usePipelineConfig() {
  return useQuery({
    queryKey: queryKeys.config,
    queryFn: api.getConfig,
    staleTime: Infinity, // server config only changes on restart
  })
}

export function useCollections() {
  return useQuery({
    queryKey: queryKeys.collections,
    queryFn: api.getCollections,
    staleTime: 60_000, // new collections appear only after an ingest
  })
}

export function useCacheStats() {
  return useQuery({
    queryKey: queryKeys.cacheStats,
    queryFn: api.getCacheStats,
  })
}

export function useClearCache() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.clearCache,
    onSuccess: (stats) => queryClient.setQueryData(queryKeys.cacheStats, stats),
  })
}

export function useHistoryList(params: HistoryParams) {
  return useQuery({
    queryKey: queryKeys.historyList(params),
    queryFn: () => api.getHistory(params),
    placeholderData: keepPreviousData, // keep the current page visible while the next loads
  })
}

export function useHistoryDetail(id: number) {
  return useQuery({
    queryKey: queryKeys.historyDetail(id),
    queryFn: async () => {
      const detail = await api.getHistoryItem(id)
      return { item: detail, run: historyToRun(detail) }
    },
    enabled: Number.isInteger(id) && id > 0,
    // Finished searches never change, so don't refetch these large payloads.
    staleTime: (query) => (query.state.data?.item.status === 'running' ? 0 : Infinity),
  })
}

export function useDeleteHistoryItem() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.deleteHistoryItem,
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: queryKeys.historyDetail(id) })
      queryClient.invalidateQueries({ queryKey: queryKeys.history })
    },
  })
}

export function useClearHistory() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.clearHistory,
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: [...queryKeys.history, 'detail'] })
      queryClient.invalidateQueries({ queryKey: queryKeys.history })
    },
  })
}

export function useUiSettings() {
  return useQuery({
    queryKey: queryKeys.uiSettings,
    queryFn: api.getUiSettings,
    staleTime: Infinity,
    placeholderData: DEFAULT_UI_SETTINGS,
  })
}

export function useSaveUiSettings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.saveUiSettings,
    onSuccess: (settings) => queryClient.setQueryData(queryKeys.uiSettings, settings),
  })
}

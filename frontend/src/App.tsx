import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router'
import { AppLayout } from '@/components/layout/AppLayout'
import { Spinner } from '@/components/ui'

// Each page is loaded on first visit, keeping the initial bundle small.
const SearchPage = lazy(() => import('@/pages/SearchPage').then((m) => ({ default: m.SearchPage })))
const ComparePage = lazy(() => import('@/pages/ComparePage').then((m) => ({ default: m.ComparePage })))
const HistoryPage = lazy(() => import('@/pages/HistoryPage').then((m) => ({ default: m.HistoryPage })))
const HistoryDetailPage = lazy(() =>
  import('@/pages/HistoryDetailPage').then((m) => ({ default: m.HistoryDetailPage })),
)
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage').then((m) => ({ default: m.NotFoundPage })))

const pageFallback = (
  <div className="flex justify-center py-24 text-slate-400">
    <Spinner className="size-8" />
  </div>
)

export default function App() {
  return (
    <Suspense fallback={pageFallback}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<SearchPage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="history/:id" element={<HistoryDetailPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

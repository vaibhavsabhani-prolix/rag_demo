import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router'
import App from './App'
import { ApiError } from './lib/api'
import { store } from './store'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Retry once on network/server errors, never on 4xx (e.g. a deleted history item).
      retry: (failureCount, error) =>
        failureCount < 1 && !(error instanceof ApiError && error.status !== undefined && error.status < 500),
      refetchOnWindowFocus: false,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </Provider>
  </StrictMode>,
)

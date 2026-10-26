# Patent Search frontend

This directory contains the React 19 and TypeScript client for the FastAPI
patent search API. Vite builds the app and proxies `/api` requests to the
backend during development.

## Pages

The route table is defined in [src/App.tsx](/Users/hepinprolix/Desktop/demo/rag_demo/frontend/src/App.tsx):

| Path | Page |
| --- | --- |
| `/` | Standard semantic search |
| `/features` | Key-feature search |
| `/feature-compare` | Key-feature comparison |
| `/compare` | Collection comparison |
| `/collections` | Collection statistics and patent lookup |
| `/history` | Search history |
| `/history/:id` | Search details and phase data |

Pages are lazy-loaded. `AppLayout` supplies the shared navigation and layout;
search and comparison pages render streaming pipeline progress as events
arrive.

## Client flow

`src/lib/api.ts` is the single API client:

1. It sends requests to `/api`.
2. It reads ordinary JSON responses for configuration, collections, history,
   settings, and cache operations.
3. It reads streaming NDJSON for search, compare, and key-feature operations.
4. It validates every response and event with the Zod schemas in
   `src/schemas`.
5. Hooks such as `useSearch`, `useCompare`, and `useKeyFeatureSearch` connect
   the client to the pages and application state.

The standard search stream is rendered by the pipeline views under
`src/features/pipeline/phases`. Results and evidence are displayed by the
components under `src/features/results` and `src/components/patent`.

## Local development

From this directory:

```bash
npm ci
npm run dev
```

The Vite server runs on <http://localhost:5173>. Its `/api` proxy targets
`http://localhost:8000`; set `API_PROXY_TARGET` to use another backend:

```bash
API_PROXY_TARGET=http://localhost:9000 npm run dev
```

The root Docker Compose development stack starts this Vite server in a
container and sets `API_PROXY_TARGET=http://web:8000`.

## Build and preview

```bash
npm run build
npm run preview
```

`npm run build` runs the TypeScript project build and creates `dist/`. The
production Dockerfile copies that directory into the FastAPI image, where
FastAPI serves the UI and keeps `/api` routes separate.

## Lint

```bash
npm run lint
```

Oxlint configuration is in [.oxlintrc.json](/Users/hepinprolix/Desktop/demo/rag_demo/frontend/.oxlintrc.json).

## Main frontend areas

| Directory | Responsibility |
| --- | --- |
| `src/pages` | Route-level page composition |
| `src/features/search` | Standard search form and pipeline tracking |
| `src/features/pipeline` | Six-phase search progress views |
| `src/features/compare` | Collection comparison |
| `src/features/feature-compare` | Key-feature comparison |
| `src/features/keyfeatures` | Key-feature search |
| `src/features/collections` | Collection and patent inspection |
| `src/features/history` | Search history and detail views |
| `src/hooks` | API/query orchestration |
| `src/lib` | API client and formatting helpers |
| `src/schemas` | Runtime API and event schemas |
| `src/store` | Redux search, compare, and UI state |
| `src/components` | Shared layout, patent, and UI components |

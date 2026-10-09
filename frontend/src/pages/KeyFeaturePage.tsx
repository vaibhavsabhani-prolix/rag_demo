import { Alert, Spinner } from '@/components/ui'
import { KeyFeatureForm } from '@/features/keyfeatures/KeyFeatureForm'
import { KeyFeatureResultsView } from '@/features/keyfeatures/KeyFeatureResultsView'
import { useKeyFeatureSearch } from '@/hooks/useKeyFeatureSearch'

export function KeyFeaturePage() {
  const {
    status,
    currentStep,
    stepName,
    extractedFeatures,
    response,
    error,
    search,
    isPending,
  } = useKeyFeatureSearch()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Independent Key Feature Search
        </h1>
        <p className="text-slate-500 max-w-3xl">
          Provide your invention problem, domain, and details. The LLM extracts 5–15 key technical features,
          generates an independent embedding for each feature, and searches Qdrant independently to keep all
          retrieval results segregated by feature.
        </p>
      </div>

      {/* 3-Input Form */}
      <KeyFeatureForm onSubmit={search} isPending={isPending} />

      {/* Running Progress State */}
      {isPending && (
        <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-6 shadow-xs">
          <div className="flex items-center gap-3">
            <Spinner className="size-5 text-indigo-600" />
            <div>
              <p className="text-sm font-semibold text-indigo-950">
                {stepName || 'Processing Key Feature Search Flow…'}
              </p>
              <p className="text-xs text-indigo-600 mt-0.5">
                {currentStep === 1 && 'Step 1/3: Analyzing problem & disclosure with LLM to extract key features…'}
                {currentStep === 2 && 'Step 2/3: Computing distinct vector embeddings for each extracted feature…'}
                {currentStep === 3 && 'Step 3/3: Performing independent vector queries against Qdrant collection…'}
              </p>
            </div>
          </div>

          {/* Real-time extracted features preview as they are parsed */}
          {extractedFeatures.length > 0 && (
            <div className="mt-4 pt-4 border-t border-indigo-100 space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-indigo-800">
                Extracted Features ({extractedFeatures.length}):
              </span>
              <ul className="space-y-1.5">
                {extractedFeatures.map((f) => (
                  <li key={f.feature_id} className="text-xs text-indigo-900 flex items-start gap-2">
                    <span className="font-mono font-bold text-indigo-600">[{f.feature_id}]</span>
                    <span>{f.feature}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Error Alert */}
      {status === 'error' && error && (
        <Alert tone="danger" title="Key Feature Search Failed">
          {error}
        </Alert>
      )}

      {/* Results View */}
      {status === 'success' && response && (
        <KeyFeatureResultsView response={response} />
      )}
    </div>
  )
}

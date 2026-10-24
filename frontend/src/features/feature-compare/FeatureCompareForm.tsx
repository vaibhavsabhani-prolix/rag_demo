  import { zodResolver } from '@hookform/resolvers/zod'
import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { Alert, Button, Checkbox, ScaleIcon, Spinner, TextField } from '@/components/ui'
import { useCollections } from '@/hooks/queries'
import { formatCompact } from '@/lib/format'
import {
  keyFeatureCompareFormSchema,
  type KeyFeatureCompareFields,
  type KeyFeatureCompareFormValues,
  type KeyFeatureItem,
} from '@/schemas/keyfeature'
import { useSeries, MAX_COMPARED } from '@/features/compare/series'

interface FeatureCompareFormProps {
  onSubmit: (values: KeyFeatureCompareFormValues) => void
  onStop: () => void
  isPending: boolean
  extractedFeatures?: KeyFeatureItem[]
  onClearFeatures?: () => void
}

export function FeatureCompareForm({
  onSubmit,
  onStop,
  isPending,
  extractedFeatures,
  onClearFeatures,
}: FeatureCompareFormProps) {
  const collections = useCollections()
  const allCollections = collections.data?.collections ?? []
  const series = useSeries(allCollections.map((c) => c.name))
  const [excluded, setExcluded] = useState<string[]>([])
  const [submitAttempted, setSubmitAttempted] = useState(false)

  const hasExtractedFeatures = Boolean(extractedFeatures && extractedFeatures.length > 0)

  const selectedCollections = allCollections
    .map((c) => c.name)
    .filter((name) => !excluded.includes(name))

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<KeyFeatureCompareFields>({
    resolver: zodResolver(keyFeatureCompareFormSchema),
    defaultValues: {
      problem:
        'Urban farms waste water and energy because irrigation schedules do not accurately match soil moisture and plant requirements.',
      invention_title: 'AI-Based Smart Irrigation System for Urban Farming',
      invention_details:
        'This invention relates to an intelligent irrigation system designed for rooftop gardens, urban farms, and small agricultural spaces. The system integrates soil moisture sensors, temperature sensors, weather data, and crop-specific information with an AI-based control algorithm. It automatically adjusts irrigation schedules and water flow according to plant requirements and environmental conditions. An IoT-enabled controller allows users to monitor water consumption, receive alerts, and manage irrigation remotely. The system can also detect abnormal water usage and identify potential irrigation failures. The invention aims to reduce water wastage, improve crop health, and support efficient urban agriculture.',
      retrieval_limit: 5000,
      top_k: 20,
      score_threshold: 0.0,
      rerank_score_threshold: 0.6,
    },
  })

  const toggleCollection = (name: string, checked: boolean) => {
    setExcluded((prev) => (checked ? prev.filter((n) => n !== name) : [...prev, name]))
  }

  const selectAllCollections = () => setExcluded([])

  const tooMany = selectedCollections.length > MAX_COMPARED
  const noneSelected = allCollections.length > 0 && selectedCollections.length === 0

  const onFormSubmit = handleSubmit(
    (values) => {
      setSubmitAttempted(false)
      if (selectedCollections.length === 0 || tooMany) return
      onSubmit({
        ...values,
        collections: selectedCollections,
        features: hasExtractedFeatures ? extractedFeatures : undefined,
      })
    },
    (formErrors) => {
      setSubmitAttempted(true)
      console.warn('Feature compare form validation errors:', formErrors)
    },
  )

  const hasFormErrors = Object.keys(errors).length > 0

  return (
    <form
      onSubmit={onFormSubmit}
      noValidate
      className="space-y-5 rounded-xl border border-slate-200 dark:border-white/10 bg-surface p-6 shadow-xs"
    >
      <div className="border-b border-slate-200 dark:border-white/10 pb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Feature-Wise Collection Comparison</h2>
        <p className="text-sm text-slate-500 dark:text-white/70">
          Extract 5–15 key technical features once with the LLM, then query and rerank candidates across multiple
          collections (e.g. 512, 2048, 4096 tokens). Compare patent recall, best scores, chunk contexts, and overlap
          feature by feature.
        </p>
      </div>

      <div className="space-y-4">
        {/* Invention Title / Tech Domain */}
        <div>
          <TextField
            id="fc_title"
            label="Invention Title or Technical Domain"
            hint="e.g., Adaptive Thermal Management for Battery Energy Storage Systems"
            placeholder="Enter a descriptive title or domain…"
            {...register('invention_title')}
            error={errors.invention_title?.message}
            disabled={isPending}
          />
        </div>

        {/* Problem to Solve */}
        <div>
          <label htmlFor="fc_problem" className="block text-sm font-medium text-slate-700 dark:text-white">
            Problem You Are Trying to Solve <span className="text-rose-500">*</span>
          </label>
          <textarea
            id="fc_problem"
            rows={2}
            className={`mt-1.5 block w-full rounded-lg border bg-surface text-slate-900 dark:text-white placeholder:text-slate-400 dark:placeholder:text-white/40 px-3.5 py-2.5 text-sm transition-colors focus:ring-2 focus:ring-indigo-500 focus:outline-none ${
              errors.problem
                ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
                : 'border-slate-300 dark:border-white/20 focus:border-indigo-500'
            }`}
            placeholder="What core engineering or technological bottleneck is addressed? (e.g. Cell degradation caused by non-uniform thermal dissipation…)"
            {...register('problem')}
            disabled={isPending}
          />
          {errors.problem && <p className="mt-1 text-xs text-rose-600 font-medium">{errors.problem.message}</p>}
        </div>

        {/* Invention Details / Disclosure */}
        <div>
          <label htmlFor="fc_details" className="block text-sm font-medium text-slate-700 dark:text-white">
            Invention Details / Specification <span className="text-rose-500">*</span>
          </label>
          <textarea
            id="fc_details"
            rows={4}
            className={`mt-1.5 block w-full rounded-lg border bg-surface text-slate-900 dark:text-white placeholder:text-slate-400 dark:placeholder:text-white/40 px-3.5 py-2.5 text-sm transition-colors focus:ring-2 focus:ring-indigo-500 focus:outline-none ${
              errors.invention_details
                ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
                : 'border-slate-300 dark:border-white/20 focus:border-indigo-500'
            }`}
            placeholder="Paste your disclosure, draft claims, or detailed technical description. The LLM extracts specific technical features from this text…"
            {...register('invention_details')}
            disabled={isPending}
          />
          {errors.invention_details && (
            <p className="mt-1 text-xs text-rose-600 font-medium">{errors.invention_details.message}</p>
          )}
        </div>

        {/* Collections to Compare */}
        <div className="rounded-lg border border-slate-200 dark:border-white/10 bg-slate-50/50 dark:bg-surface p-3.5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 dark:text-white/80">
              Collections to Compare ({selectedCollections.length}/{allCollections.length})
            </span>
            {excluded.length > 0 && !isPending && (
              <button
                type="button"
                onClick={selectAllCollections}
                className="text-xs font-medium text-indigo-600 hover:underline dark:text-indigo-300"
              >
                Select all
              </button>
            )}
          </div>

          {collections.isPending && (
            <div className="flex items-center gap-2 py-2 text-xs text-slate-500">
              <Spinner className="size-3.5" /> Loading collections…
            </div>
          )}

          {collections.error && <Alert tone="danger">{collections.error.message}</Alert>}

          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-1">
            {allCollections.map((c, i) => (
              <span key={c.name} className="inline-flex items-center gap-2">
                <span
                  className="size-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: series[i]?.color ?? '#6366f1' }}
                  aria-hidden
                />
                <Checkbox
                  label={`${c.name} · ${formatCompact(c.chunk_count)} chunks`}
                  checked={!excluded.includes(c.name)}
                  onChange={(e) => toggleCollection(c.name, e.target.checked)}
                  disabled={isPending}
                />
              </span>
            ))}
          </div>

          {tooMany && (
            <p className="text-xs font-medium text-rose-600">
              Select at most {MAX_COMPARED} collections.
            </p>
          )}
          {noneSelected && (
            <p className="text-xs font-medium text-rose-600">
              Please select at least one collection to compare.
            </p>
          )}
        </div>

        {/* Options: Candidate Limit, Top-K Reranked, Initial Threshold, Rerank Threshold */}
        <div className="grid grid-cols-1 gap-4 pt-1 sm:grid-cols-2 lg:grid-cols-4">
          <TextField
            id="fc_retrieval_limit"
            type="number"
            label="Candidate Limit"
            hint="Chunks retrieved per feature"
            min={10}
            max={10000}
            step={100}
            placeholder="5000"
            {...register('retrieval_limit', { valueAsNumber: true })}
            error={errors.retrieval_limit?.message}
            disabled={isPending}
          />

          <TextField
            id="fc_top_k"
            type="number"
            label="Top Patents (Reranked)"
            hint="Final patents kept per feature"
            min={1}
            max={200}
            step={1}
            placeholder="20"
            {...register('top_k', { valueAsNumber: true })}
            error={errors.top_k?.message}
            disabled={isPending}
          />

          <TextField
            id="fc_score_threshold"
            type="number"
            label="Initial Threshold"
            hint="Min vector score (0.0 for all)"
            min={0}
            max={1}
            step={0.01}
            placeholder="0.0"
            {...register('score_threshold', { valueAsNumber: true })}
            error={errors.score_threshold?.message}
            disabled={isPending}
          />

          <TextField
            id="fc_rerank_score_threshold"
            type="number"
            label="Rerank Threshold"
            hint="Min cross-encoder score"
            min={0}
            max={1}
            step={0.01}
            placeholder="0.6"
            {...register('rerank_score_threshold', { valueAsNumber: true })}
            error={errors.rerank_score_threshold?.message}
            disabled={isPending}
          />
        </div>
      </div>

      {/* Validation feedback reminder if user clicked with missing required fields */}
      {submitAttempted && (hasFormErrors || noneSelected) && (
        <div className="rounded-lg bg-rose-50 p-3 text-xs text-rose-700 border border-rose-200">
          <p className="font-semibold">Please fix the following before starting comparison:</p>
          <ul className="mt-1 list-disc list-inside space-y-0.5">
            {errors.problem && <li>Problem: {errors.problem.message}</li>}
            {errors.invention_title && <li>Title: {errors.invention_title.message}</li>}
            {errors.invention_details && <li>Details: {errors.invention_details.message}</li>}
            {noneSelected && <li>Collections: Select at least one collection</li>}
          </ul>
        </div>
      )}

      {/* Feature Reuse Status Banner (LLM Extraction Runs Once) */}
      {hasExtractedFeatures && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-emerald-500/40 bg-emerald-950/30 p-3.5 text-xs">
          <div className="flex items-center gap-2.5">
            <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-white text-2xs font-bold shadow-xs">
              ✓
            </span>
            <div>
              <span className="font-semibold text-white text-sm">
                {extractedFeatures?.length} Key Technical Features Extracted
              </span>
              <p className="text-white/80 text-xs mt-0.5">
                Features are saved and will be reused directly. Comparing collections will NOT call the LLM again.
              </p>
            </div>
          </div>
          {onClearFeatures && !isPending && (
            <button
              type="button"
              onClick={onClearFeatures}
              className="text-xs font-semibold text-emerald-300 hover:text-white underline transition-colors"
              title="Click if you modified the invention details and want fresh features from the LLM"
            >
              Re-extract with LLM
            </button>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-wrap items-center justify-end gap-3 pt-3 border-t border-slate-200 dark:border-white/10">
        {isPending && (
          <Button variant="secondary" size="lg" onClick={onStop} className="w-full sm:w-auto">
            Stop Comparison
          </Button>
        )}
        <Button
          type="submit"
          size="lg"
          loading={isPending}
          disabled={noneSelected || tooMany}
          className="w-full sm:w-auto gap-2"
        >
          {!isPending && <ScaleIcon className="size-4" />}
          {isPending
            ? 'Comparing Features Across Collections…'
            : hasExtractedFeatures
            ? `Compare Collections (Reusing ${extractedFeatures?.length} Features · LLM Skipped)`
            : 'Extract Features & Compare Collections'}
        </Button>
      </div>
    </form>
  )
}

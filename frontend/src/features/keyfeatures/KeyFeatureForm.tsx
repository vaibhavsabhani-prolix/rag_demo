import { zodResolver } from '@hookform/resolvers/zod'
import { useEffect } from 'react'
import { useForm, useWatch } from 'react-hook-form'
import { Button, SelectField, TextField, SparklesIcon } from '@/components/ui'
import { useCollections } from '@/hooks/queries'
import { formatCompact } from '@/lib/format'
import { keyFeatureFormSchema, type KeyFeatureFormValues } from '@/schemas/keyfeature'

interface KeyFeatureFormProps {
  onSubmit: (values: KeyFeatureFormValues) => void
  isPending: boolean
}

export function KeyFeatureForm({ onSubmit, isPending }: KeyFeatureFormProps) {
  const collections = useCollections()

  const {
    register,
    handleSubmit,
    setValue,
    control,
    formState: { errors },
  } = useForm<KeyFeatureFormValues>({
    resolver: zodResolver(keyFeatureFormSchema),
    defaultValues: {
      problem:
        'Urban farms waste water and energy because irrigation schedules do not accurately match soil moisture and plant requirements.',
      invention_title: 'AI-Based Smart Irrigation System for Urban Farming',
      invention_details:
        'This invention relates to an intelligent irrigation system designed for rooftop gardens, urban farms, and small agricultural spaces. The system integrates soil moisture sensors, temperature sensors, weather data, and crop-specific information with an AI-based control algorithm. It automatically adjusts irrigation schedules and water flow according to plant requirements and environmental conditions. An IoT-enabled controller allows users to monitor water consumption, receive alerts, and manage irrigation remotely. The system can also detect abnormal water usage and identify potential irrigation failures. The invention aims to reduce water wastage, improve crop health, and support efficient urban agriculture.',
      collection: collections.data?.default ?? '512',
      retrieval_limit: 5000,
      top_k: 20,
      score_threshold: 0.0,
      rerank_score_threshold: 0.6,
    },
  })

  // Keep collection default in sync
  const selectedColl = useWatch({ control, name: 'collection' })
  useEffect(() => {
    const data = collections.data
    if (!data || data.collections.some((c) => c.name === selectedColl)) return
    const next = data.default ?? data.collections[0]?.name ?? ''
    if (next && next !== selectedColl) setValue('collection', next)
  }, [collections.data, selectedColl, setValue])

  const noCollections = collections.data?.collections.length === 0

  const onFormSubmit = handleSubmit((values) => {
    onSubmit(values)
  })

  return (
    <form onSubmit={onFormSubmit} noValidate className="space-y-5 rounded-xl border border-slate-200 bg-surface p-6 shadow-xs">
      <div className="border-b border-slate-100 pb-4">
        <h2 className="text-lg font-semibold text-slate-900">Key Feature Search Flow</h2>
        <p className="text-sm text-slate-500">
          Extracts 5–15 key technical features via LLM, embeds &amp; retrieves candidate chunks from Qdrant, reranks them with BGE Cross-Encoder, and returns the top chunks per feature.
        </p>
      </div>

      <div className="space-y-4">
        {/* Input 1: Problem */}
        <div>
          <label htmlFor="problem" className="mb-1 block text-sm font-semibold text-slate-800">
            1. Problem you are trying to solve <span className="text-rose-500">*</span>
          </label>
          <textarea
            id="problem"
            {...register('problem')}
            rows={2}
            placeholder="e.g. High energy consumption in wireless sensor networks requiring frequent battery replacement..."
            className={`block w-full rounded-lg border bg-surface p-3 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 ${
              errors.problem
                ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
                : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-100'
            }`}
          />
          {errors.problem && (
            <p className="mt-1 text-xs text-rose-600">{errors.problem.message}</p>
          )}
        </div>

        {/* Input 2: Title / Domain */}
        <div>
          <label htmlFor="invention_title" className="mb-1 block text-sm font-semibold text-slate-800">
            2. Invention title / technology domain <span className="text-rose-500">*</span>
          </label>
          <TextField
            id="invention_title"
            {...register('invention_title')}
            placeholder="e.g. Ultra-Low Power Bluetooth Beacon with Energy Harvesting"
            error={errors.invention_title?.message}
          />
        </div>

        {/* Input 3: Invention Details / Disclosure */}
        <div>
          <label htmlFor="invention_details" className="mb-1 block text-sm font-semibold text-slate-800">
            3. Invention details / disclosure <span className="text-rose-500">*</span>
          </label>
          <textarea
            id="invention_details"
            {...register('invention_details')}
            rows={4}
            placeholder="Provide technical description, components, mechanisms, and operational steps of your invention..."
            className={`block w-full rounded-lg border bg-surface p-3 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-4 ${
              errors.invention_details
                ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100'
                : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-100'
            }`}
          />
          {errors.invention_details && (
            <p className="mt-1 text-xs text-rose-600">{errors.invention_details.message}</p>
          )}
        </div>

        {/* Options: Collection, Retrieval Limit, Top-K Reranked, Initial Threshold, & Rerank Threshold */}
        <div className="grid grid-cols-1 gap-4 pt-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          <SelectField
            id="kf_collection"
            label="Qdrant Collection"
            {...register('collection')}
            disabled={!collections.data || noCollections}
            error={
              collections.error?.message ??
              (noCollections ? 'No searchable collection in Qdrant.' : errors.collection?.message)
            }
          >
            {!collections.data && <option value="">{collections.error ? 'Unavailable' : 'Loading…'}</option>}
            {noCollections && <option value="">No collections</option>}
            {collections.data?.collections.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({formatCompact(c.chunk_count)} chunks)
              </option>
            ))}
          </SelectField>

          <TextField
            id="kf_retrieval_limit"
            type="number"
            label="Candidate Limit"
            hint="Chunks retrieved per feature (default 5000)"
            min={10}
            max={10000}
            step={100}
            placeholder="5000"
            {...register('retrieval_limit', { valueAsNumber: true })}
            error={errors.retrieval_limit?.message}
          />

          <TextField
            id="kf_top_k"
            type="number"
            label="Top Patents (Reranked)"
            hint="Final top patents kept per feature (default 20)"
            min={1}
            max={200}
            step={1}
            placeholder="20"
            {...register('top_k', { valueAsNumber: true })}
            error={errors.top_k?.message}
          />

          <TextField
            id="kf_score_threshold"
            type="number"
            label="Initial Threshold"
            hint="Min vector score for candidates (0.0 for all)"
            min={0}
            max={1}
            step={0.01}
            placeholder="0.0"
            {...register('score_threshold', { valueAsNumber: true })}
            error={errors.score_threshold?.message}
          />

          <TextField
            id="kf_rerank_score_threshold"
            type="number"
            label="Rerank Threshold"
            hint="Min cross-encoder score to keep (default 0.6)"
            min={0}
            max={1}
            step={0.01}
            placeholder="0.6"
            {...register('rerank_score_threshold', { valueAsNumber: true })}
            error={errors.rerank_score_threshold?.message}
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-3 pt-2">
        <Button
          type="submit"
          size="lg"
          loading={isPending}
          className="min-w-44"
        >
          <SparklesIcon className="size-4" />
          {isPending ? 'Processing Flow…' : 'Extract & Search Features'}
        </Button>
      </div>
    </form>
  )
}

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
      problem: '',
      invention_title: '',
      invention_details: '',
      collection: collections.data?.default ?? '512',
      top_k: 20,
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
          Extracts 5–15 key technical features via LLM, then embeds and searches Qdrant for each feature independently.
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

        {/* Options: Collection & Top-K */}
        <div className="grid grid-cols-1 gap-4 pt-2 sm:grid-cols-2">
          <SelectField
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
            type="number"
            label="Chunks per Feature (Top-K)"
            min={1}
            max={200}
            {...register('top_k', { valueAsNumber: true })}
            error={errors.top_k?.message}
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

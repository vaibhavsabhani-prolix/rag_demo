import { zodResolver } from '@hookform/resolvers/zod'
import { useEffect } from 'react'
import { useForm, useWatch } from 'react-hook-form'
import { useSearchParams } from 'react-router'
import { Button, Checkbox, SearchIcon, SelectField, TextField } from '@/components/ui'
import { useCollections } from '@/hooks/queries'
import { useSearch } from '@/hooks/useSearch'
import { formatCompact } from '@/lib/format'
import { searchFormSchema, type SearchFormValues } from '@/schemas/search'

const EXAMPLES = [
  'Automotive rear-view mirror with an integrated display',
  'Solid-state battery with a sulfide electrolyte layer',
  'Foldable smartphone hinge using a flexible OLED panel',
]

// Remembers the last searched collection in this browser.
const COLLECTION_STORAGE_KEY = 'patent-search.collection'

function loadSavedCollection(): string | null {
  try {
    return localStorage.getItem(COLLECTION_STORAGE_KEY)
  } catch {
    return null
  }
}

function saveCollection(name: string) {
  try {
    localStorage.setItem(COLLECTION_STORAGE_KEY, name)
  } catch {
    // storage unavailable; the server default is used next time
  }
}

export function SearchForm() {
  const search = useSearch()
  const collections = useCollections()
  // "Search again" from history links here with ?q=…&collection=… to prefill the form.
  const [params] = useSearchParams()
  const {
    register,
    handleSubmit,
    setValue,
    control,
    formState: { errors },
  } = useForm<SearchFormValues>({
    resolver: zodResolver(searchFormSchema),
    defaultValues: {
      query: params.get('q') ?? '',
      collection: params.get('collection') ?? loadSavedCollection() ?? '',
      useCache: true,
    },
  })

  // Fall back to the server's default when the chosen collection doesn't exist (any more).
  const selected = useWatch({ control, name: 'collection' })
  useEffect(() => {
    const data = collections.data
    if (!data || data.collections.some((c) => c.name === selected)) return
    const next = data.default ?? ''
    if (next !== selected) setValue('collection', next)
  }, [collections.data, selected, setValue])

  const onSubmit = handleSubmit((values) => {
    saveCollection(values.collection)
    search.mutate(values)
  })

  const noCollections = collections.data?.collections.length === 0

  return (
    <form onSubmit={onSubmit} noValidate className="space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <TextField
          {...register('query')}
          aria-label="Patent search query"
          placeholder="Describe the invention you're looking for…"
          leading={<SearchIcon className="size-5" />}
          error={errors.query?.message}
          className="flex-1"
          autoFocus
        />
        <SelectField
          {...register('collection')}
          aria-label="Collection to search"
          title="Collection to search"
          disabled={!collections.data || noCollections}
          error={
            collections.error?.message ??
            (noCollections ? 'No searchable collection in Qdrant.' : errors.collection?.message)
          }
          className="sm:w-52"
        >
          {!collections.data && <option value="">{collections.error ? 'Unavailable' : 'Loading…'}</option>}
          {noCollections && <option value="">No collections</option>}
          {collections.data?.collections.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} · {formatCompact(c.chunk_count)} chunks
            </option>
          ))}
        </SelectField>
        <Button type="submit" size="lg" loading={search.isPending} className="sm:w-36">
          {search.isPending ? 'Searching' : 'Search'}
        </Button>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-slate-400">Try:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => setValue('query', example, { shouldValidate: true })}
              className="rounded-full bg-surface px-3 py-1 text-slate-600 ring-1 ring-slate-200 hover:text-indigo-600 hover:ring-indigo-300"
            >
              {example}
            </button>
          ))}
        </div>
        <Checkbox {...register('useCache')} label="Use query cache" />
      </div>
    </form>
  )
}

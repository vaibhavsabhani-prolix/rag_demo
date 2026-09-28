import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { useSearchParams } from 'react-router'
import { Button, Checkbox, SearchIcon, TextField } from '@/components/ui'
import { useSearch } from '@/hooks/useSearch'
import { searchFormSchema, type SearchFormValues } from '@/schemas/search'

const EXAMPLES = [
  'Automotive rear-view mirror with an integrated display',
  'Solid-state battery with a sulfide electrolyte layer',
  'Foldable smartphone hinge using a flexible OLED panel',
]

export function SearchForm() {
  const search = useSearch()
  // "Search again" from history links here with ?q=… to prefill the box.
  const [params] = useSearchParams()
  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<SearchFormValues>({
    resolver: zodResolver(searchFormSchema),
    defaultValues: { query: params.get('q') ?? '', useCache: true },
  })

  const onSubmit = handleSubmit((values) => search.mutate(values))

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

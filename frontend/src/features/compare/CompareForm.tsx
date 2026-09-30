import { zodResolver } from '@hookform/resolvers/zod'
import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { Alert, Button, Checkbox, SearchIcon, Spinner, TextField } from '@/components/ui'
import { useCollections } from '@/hooks/queries'
import type { CompareController } from '@/hooks/useCompare'
import { formatCompact } from '@/lib/format'
import { searchFormSchema } from '@/schemas/search'
import { MAX_COMPARED, useSeries } from './series'

const compareFormSchema = searchFormSchema.pick({ query: true })

export function CompareForm({ compare }: { compare: CompareController }) {
  const collections = useCollections()
  // Tracks what was unticked, so collections added later are included by default.
  const [excluded, setExcluded] = useState<string[]>([])
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(compareFormSchema),
    defaultValues: { query: '' },
  })

  const all = collections.data?.collections ?? []
  const selected = all.map((c) => c.name).filter((name) => !excluded.includes(name))
  const series = useSeries(all.map((c) => c.name))
  const tooMany = selected.length > MAX_COMPARED
  const running = compare.isPending

  const toggle = (name: string, checked: boolean) =>
    setExcluded((prev) => (checked ? prev.filter((n) => n !== name) : [...prev, name]))

  const submit = (oneByOne: boolean) =>
    handleSubmit(({ query }) => {
      if (selected.length === 0 || tooMany) return
      compare.start({ query, collections: selected, oneByOne })
    })

  return (
    <form onSubmit={submit(false)} noValidate className="space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <TextField
          {...register('query')}
          aria-label="Query to compare"
          placeholder="Describe the invention to search every collection for…"
          leading={<SearchIcon className="size-5" />}
          error={errors.query?.message}
          className="flex-1"
          autoFocus
        />
        {running ? (
          <Button variant="secondary" size="lg" onClick={compare.stop} className="sm:w-36">
            <Spinner className="size-4" />
            Stop
          </Button>
        ) : (
          <div className="flex gap-3">
            <Button
              type="submit"
              size="lg"
              disabled={selected.length === 0 || tooMany}
              className="flex-1 sm:w-36"
              title="Run every selected collection in a row"
            >
              Compare all
            </Button>
            <Button
              variant="secondary"
              size="lg"
              disabled={selected.length === 0 || tooMany}
              onClick={submit(true)}
              className="flex-1 sm:w-36"
              title="Run the first collection now, then choose which to run next"
            >
              One by one
            </Button>
          </div>
        )}
      </div>

      {collections.error && <Alert tone="danger">{collections.error.message}</Alert>}

      <fieldset className="flex flex-wrap items-center gap-x-4 gap-y-2" disabled={running}>
        <legend className="sr-only">Collections to compare</legend>
        <span className="text-sm text-slate-400">Collections:</span>
        {collections.isPending && <Spinner className="size-4 text-slate-400" />}
        {collections.data && all.length === 0 && (
          <span className="text-sm text-slate-500">No searchable collection in Qdrant.</span>
        )}
        {all.map((c, i) => (
          <span key={c.name} className="inline-flex items-center gap-1.5">
            <span className="size-2.5 rounded-sm" style={{ backgroundColor: series[i].color }} aria-hidden />
            <Checkbox
              label={`${c.name} · ${formatCompact(c.chunk_count)} chunks`}
              checked={!excluded.includes(c.name)}
              onChange={(e) => toggle(c.name, e.target.checked)}
            />
          </span>
        ))}
      </fieldset>

      {tooMany && <p className="text-sm text-rose-600">Select at most {MAX_COMPARED} collections.</p>}
      {collections.data && all.length > 0 && selected.length === 0 && (
        <p className="text-sm text-rose-600">Select at least one collection.</p>
      )}
    </form>
  )
}

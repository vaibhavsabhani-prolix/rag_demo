import { useState } from 'react'
import { Alert, Button, Card, CardBody, CardHeader, SearchIcon, TextField } from '@/components/ui'
import { ApiError } from '@/lib/api'
import { useLookupPatentStorage } from '@/hooks/queries'
import { SamplePatentComparison } from './SamplePatentComparison'

/** Look up any patent_id's chunk count and storage footprint across every collection, on demand. */
export function PatentLookup() {
  const [patentId, setPatentId] = useState('')
  const lookup = useLookupPatentStorage()

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = patentId.trim()
    if (trimmed) lookup.mutate(trimmed)
  }

  const result = lookup.data

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Look up a patent"
          subtitle="Enter a patent ID to see its chunk count and storage footprint across every collection."
        />
        <CardBody className="space-y-4">
          <form onSubmit={onSubmit} className="flex items-end gap-3">
            <TextField
              label="Patent ID"
              placeholder="e.g. CN108282018B"
              value={patentId}
              onChange={(e) => setPatentId(e.target.value)}
              leading={<SearchIcon className="size-4" />}
              className="max-w-xs"
            />
            <Button type="submit" loading={lookup.isPending} disabled={!patentId.trim()}>
              Look up
            </Button>
          </form>

          {lookup.isError && (
            <Alert tone="danger" title="Lookup failed">
              {lookup.error instanceof ApiError ? lookup.error.message : 'Something went wrong.'}
            </Alert>
          )}

          {result && !result.found && (
            <Alert tone="warning" title={`"${result.patent_id}" was not found`}>
              It has no metadata in any searchable collection ({result.missing_collections.join(', ')}). Check the
              ID and try again.
            </Alert>
          )}

          {result && result.found && result.missing_collections.length > 0 && (
            <Alert tone="info" title="Only partially indexed">
              "{result.patent_id}" was not found in: {result.missing_collections.join(', ')}. Showing the
              collections it is in.
            </Alert>
          )}
        </CardBody>
      </Card>

      {result && result.found && (
        <SamplePatentComparison
          patent={{ patent_id: result.patent_id, metadata: result.metadata, per_collection: result.per_collection }}
        />
      )}
    </div>
  )
}

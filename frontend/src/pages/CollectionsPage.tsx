import { Alert, DatabaseIcon, EmptyState, Spinner } from '@/components/ui'
import { CollectionStorageTable } from '@/features/collections/CollectionStorageTable'
import { PatentLookup } from '@/features/collections/PatentLookup'
import { SamplePatentComparison } from '@/features/collections/SamplePatentComparison'
import { useCollectionsOverview } from '@/hooks/queries'

export function CollectionsPage() {
  const overview = useCollectionsOverview()

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Collections</h1>
        <p className="text-slate-500">
          Storage and memory usage for each chunk-size collection, and the same sample patent compared across them —
          how chunking it at a smaller or larger token limit changes how many chunks, and how much storage, it takes.
        </p>
      </div>

      {overview.isLoading && (
        <div className="flex justify-center py-16 text-slate-400">
          <Spinner className="size-8" />
        </div>
      )}

      {overview.isError && (
        <Alert tone="danger" title="Could not load collection stats">
          {overview.error.message}
        </Alert>
      )}

      {overview.data && overview.data.collections.length === 0 && (
        <EmptyState
          icon={<DatabaseIcon className="size-10" />}
          title="No collections indexed yet"
          description="Run the ingestion pipeline to create a searchable collection."
        />
      )}

      {overview.data && overview.data.collections.length > 0 && (
        <>
          <CollectionStorageTable collections={overview.data.collections} />

          <div className="space-y-2">
            <h2 className="text-lg font-semibold text-slate-900">Sample patent, across collections</h2>
            <p className="text-sm text-slate-500">
              {overview.data.sample_patents.length > 0
                ? 'The same patent(s), found in every collection, so chunk count and storage can be compared head-to-head.'
                : 'No patent is indexed in every collection yet, so there is nothing to compare side by side.'}
            </p>
          </div>

          {overview.data.sample_patents.map((patent) => (
            <SamplePatentComparison key={patent.patent_id} patent={patent} />
          ))}

          <PatentLookup />
        </>
      )}
    </div>
  )
}

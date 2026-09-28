import { ExternalIcon } from '@/components/ui'
import { usePipelineConfig } from '@/hooks/queries'
import { patentUrl, summarizePatent } from '@/lib/patent'

interface PatentHeadingProps {
  patentId: string
  metadata: Record<string, unknown>
  rank?: number
}

/** Patent id (linked to the patent viewer), title, assignee, country and year. */
export function PatentHeading({ patentId, metadata, rank }: PatentHeadingProps) {
  const { data: config } = usePipelineConfig()
  const { title, assignee, year, country } = summarizePatent(metadata)
  const url = patentUrl(config?.patent_view_url_template, patentId)

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        {rank !== undefined && <span className="font-semibold text-slate-400">#{rank}</span>}
        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="inline-flex items-center gap-1 font-mono font-semibold text-indigo-600 hover:underline"
          >
            {patentId}
            <ExternalIcon className="size-3.5" />
          </a>
        ) : (
          <span className="font-mono font-semibold text-indigo-600">{patentId}</span>
        )}
        <span className="text-slate-400">·</span>
        <span className="text-slate-500">
          {country} · {year}
        </span>
      </div>
      {title && <h3 className="mt-1 leading-snug font-semibold text-slate-900">{title}</h3>}
      <p className="mt-0.5 truncate text-sm text-slate-500">{assignee}</p>
    </div>
  )
}

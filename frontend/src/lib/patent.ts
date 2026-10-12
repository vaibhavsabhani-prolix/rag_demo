/** Helpers for reading the loosely-typed patent metadata dictionary. */
import { formatValue } from './format'

type Metadata = Record<string, unknown>

function first(meta: Metadata, keys: string[]): unknown {
  for (const key of keys) {
    const value = meta[key]
    if (value !== undefined && value !== null && value !== '') return value
  }
  return undefined
}

export interface PatentSummary {
  title: string
  assignee: string
  year: string
  country: string
}

export function summarizePatent(meta: Metadata): PatentSummary {
  const title = first(meta, ['Title-english', 'Title'])
  const assignee = first(meta, ['Current Assignee Standardized', 'Applicant First Organization'])
  return {
    title: title ? formatValue(title) : '',
    assignee: assignee ? formatValue(assignee) : 'Unknown assignee',
    year: formatValue(first(meta, ['Publication Year', 'Application Year'])),
    country: formatValue(first(meta, ['Publication Country Code'])),
  }
}

export const DEFAULT_PATENT_VIEW_URL_TEMPLATE =
  'https://www.qubeip.com/en/patent-view/{patent_id}'

export function patentUrl(template: string | undefined, patentId: string): string {
  const tmpl = template || DEFAULT_PATENT_VIEW_URL_TEMPLATE
  return tmpl.replace('{patent_id}', encodeURIComponent(patentId))
}

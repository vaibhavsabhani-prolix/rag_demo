/** Chart colour per collection. */
import { useCollections } from '@/hooks/queries'
import type { Series } from './charts'

/** The palette has 8 validated colours, so at most 8 collections are compared at once. */
export const MAX_COMPARED = 8

/**
 * Colours follow the collection, not its position in this comparison: each
 * collection keeps its slot from the full, sorted collection list, so leaving
 * one out never repaints the others.
 */
export function useSeries(names: string[]): Series[] {
  const all = useCollections().data?.collections.map((c) => c.name) ?? []
  return names.map((name, i) => {
    const slot = all.indexOf(name)
    return { name, color: `var(--series-${((slot === -1 ? i : slot) % MAX_COMPARED) + 1})` }
  })
}

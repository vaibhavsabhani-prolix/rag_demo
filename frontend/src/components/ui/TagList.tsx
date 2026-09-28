import type { Tone } from '@/lib/tone'
import { Badge } from './Badge'

interface TagListProps {
  items: string[]
  tone?: Tone
  emptyText?: string
}

/** A wrapping row of badges, e.g. concepts or matched views. */
export function TagList({ items, tone = 'neutral', emptyText = 'None' }: TagListProps) {
  if (items.length === 0) return <span className="text-sm text-slate-400">{emptyText}</span>
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <Badge key={item} tone={tone}>
          {item}
        </Badge>
      ))}
    </div>
  )
}

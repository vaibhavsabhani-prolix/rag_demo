interface BulletListProps {
  items: string[]
  emptyText?: string
}

export function BulletList({ items, emptyText = 'None' }: BulletListProps) {
  if (items.length === 0) return <span className="text-sm text-slate-400">{emptyText}</span>
  return (
    <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700 marker:text-slate-300">
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  )
}

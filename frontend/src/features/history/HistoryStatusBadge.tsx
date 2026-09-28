import { Badge } from '@/components/ui'
import type { HistoryItem } from '@/schemas/history'

const LABELS: Record<HistoryItem['status'], string> = {
  success: 'Completed',
  error: 'Failed',
  running: 'Interrupted',
}

export function HistoryStatusBadge({ status }: { status: HistoryItem['status'] }) {
  return <Badge tone={status === 'success' ? 'success' : status === 'error' ? 'danger' : 'warning'}>{LABELS[status]}</Badge>
}

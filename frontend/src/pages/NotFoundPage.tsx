import { Link } from 'react-router'
import { EmptyState } from '@/components/ui'

export function NotFoundPage() {
  return (
    <EmptyState
      title="Page not found"
      description={
        <Link to="/" className="font-medium text-indigo-600 hover:underline">
          Go to search
        </Link>
      }
    />
  )
}

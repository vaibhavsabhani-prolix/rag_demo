import clsx from 'clsx'
import { finalScoreTone, type Tone } from '@/lib/tone'

const styles: Partial<Record<Tone, string>> = {
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  info: 'bg-sky-50 text-sky-700 ring-sky-200',
  brand: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
  neutral: 'bg-slate-50 text-slate-500 ring-slate-200',
}

/** Large final-score pill (0–10 scale). null for a metadata-only query, which isn't scored. */
export function ScoreBadge({ score }: { score: number | null }) {
  return (
    <div className={clsx('rounded-lg px-3 py-1.5 text-center ring-1', styles[finalScoreTone(score)])}>
      <div className="text-xl leading-none font-bold tabular-nums">{score === null ? '—' : score.toFixed(2)}</div>
      <div className="mt-0.5 text-[10px] font-medium tracking-wide uppercase opacity-70">
        {score === null ? 'not scored' : 'of 10'}
      </div>
    </div>
  )
}

import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card, CardBody, CardHeader, SegmentedControl } from '@/components/ui'
import type { KeyFeatureItem, KeyFeatureSearchResponse } from '@/schemas/keyfeature'
import type { Series } from '@/features/compare/charts'

interface FeatureScoreChartProps {
  features: KeyFeatureItem[]
  runs: Record<string, { response?: KeyFeatureSearchResponse }>
  series: Series[]
}

export function FeatureScoreChart({ features, runs, series }: FeatureScoreChartProps) {
  const [metric, setMetric] = useState<'score' | 'count'>('score')

  // Prepare chart data
  const data = features.map((feat) => {
    const row: Record<string, unknown> = {
      feature_id: feat.feature_id,
      label: `F${feat.feature_id}`,
      featureText: feat.feature,
    }

    for (const s of series) {
      const resp = runs[s.name]?.response
      const featRes = resp?.results?.find((r) => r.feature_id === feat.feature_id)

      if (metric === 'score') {
        let best = 0
        if (featRes) {
          for (const p of featRes.patents || []) {
            const sc = p.best_reranker_score ?? p.score ?? 0
            if (sc > best) best = sc
          }
          if (best === 0 && featRes.chunks && featRes.chunks.length > 0) {
            for (const c of featRes.chunks) {
              const sc = c.reranker_score ?? c.score ?? 0
              if (sc > best) best = sc
            }
          }
        }
        row[s.name] = best > 0 ? Number(best.toFixed(3)) : 0
      } else {
        // Patent count
        const count = featRes?.patents?.length ?? new Set(featRes?.chunks?.map((c) => c.patent_id) || []).size
        row[s.name] = count
      }
    }
    return row
  })

  return (
    <Card>
      <CardHeader
        title={metric === 'score' ? 'Top Rerank Score per Feature' : 'Matching Patents per Feature'}
        subtitle={
          metric === 'score'
            ? 'Highest cross-encoder confidence score found in each collection for each key feature.'
            : 'Number of distinct candidate patents found in each collection for each key feature.'
        }
        actions={
          <SegmentedControl
            label="Metric"
            value={metric}
            onChange={(val) => setMetric(val as 'score' | 'count')}
            options={[
              { value: 'score', label: 'Best Score' },
              { value: 'count', label: 'Patent Count' },
            ]}
          />
        }
      />
      <CardBody>
        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-slate-200, #e2e8f0)" />
              <XAxis
                dataKey="label"
                tick={{ fill: 'var(--color-slate-600, #475569)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--color-slate-300, #cbd5e1)' }}
              />
              <YAxis
                domain={metric === 'score' ? [0, 1] : ['auto', 'auto']}
                tick={{ fill: 'var(--color-slate-600, #475569)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--color-slate-300, #cbd5e1)' }}
                tickFormatter={(v) => (metric === 'score' ? v.toFixed(2) : String(v))}
              />
              <Tooltip
                cursor={{ fill: 'var(--color-slate-100)' }}
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  const item = data.find((d) => d.label === label)
                  return (
                    <div className="rounded-lg border border-slate-200 dark:border-white/10 bg-surface p-3 shadow-md max-w-sm text-xs">
                      <p className="font-semibold text-slate-900 dark:text-white">
                        Feature {String(item?.feature_id)}: {String(item?.featureText || '')}
                      </p>
                      <div className="mt-2 space-y-1">
                        {payload.map((entry) => (
                          <div key={entry.name} className="flex items-center justify-between gap-4">
                            <span className="flex items-center gap-1.5 text-slate-600 dark:text-white/80">
                              <span
                                className="size-2.5 rounded-sm"
                                style={{ backgroundColor: entry.color }}
                              />
                              {entry.name}:
                            </span>
                            <span className="font-mono font-bold text-slate-900 dark:text-white">
                              {metric === 'score'
                                ? Number(entry.value).toFixed(3)
                                : `${entry.value} patents`}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                }}
              />
              <Legend
                verticalAlign="top"
                align="right"
                wrapperStyle={{ paddingBottom: '8px', fontSize: '12px', color: 'var(--color-slate-700)' }}
              />
              {series.map((s) => (
                <Bar
                  key={s.name}
                  dataKey={s.name}
                  name={s.name}
                  fill={s.color}
                  radius={[3, 3, 0, 0]}
                  maxBarSize={36}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardBody>
    </Card>
  )
}

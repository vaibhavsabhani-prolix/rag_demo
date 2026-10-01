import { useState, type ReactNode } from 'react'
import { Card, CardBody, CardHeader, DataTable, Stat, Tabs, type Column } from '@/components/ui'
import { SearchRunView } from '@/features/run/SearchRunView'
import { formatBytes, formatMs } from '@/lib/format'
import { PHASES, type PhaseNumber } from '@/schemas/pipeline'
import type { CompareState } from '@/store/compareSlice'
import { CollectionBarChart, GroupedBarChart, Legend, RankLineChart, type Series } from './charts'
import {
  COLLECTION_PHASES,
  TOP_CANDIDATES,
  keyFindings,
  resultOverlap,
  sharedCandidates,
  type CollectionMetrics,
  type Finding,
  type OverlapRow,
} from './metrics'

const formatCount = (n: number) => n.toLocaleString()
const formatScore2 = (n: number) => n.toFixed(2)
/** Milliseconds as seconds, for chart axes and labels. */
const formatSeconds = (ms: number) => `${(ms / 1000).toFixed(ms > 0 && ms < 10_000 ? 1 : 0)} s`
/** Megabytes, for chart axes and labels (charts get MB so their ticks come out round). */
const formatMB = (mb: number) => `${mb.toFixed(mb > 0 && mb < 10 ? 1 : 0)} MB`
const toMB = (bytes: number | undefined) => (bytes === undefined ? undefined : bytes / 2 ** 20)
const orDash = <T,>(v: T | undefined, format: (v: T) => string) => (v === undefined ? '—' : format(v))

interface CompareDashboardProps {
  state: CompareState
  metrics: CollectionMetrics[]
  series: Series[]
}

/** Charts and tables comparing each collection's run of the same query. */
export function CompareDashboard({ state, metrics, series }: CompareDashboardProps) {
  const byName = Object.fromEntries(metrics.map((m) => [m.name, m]))
  const threshold = metrics.find((m) => m.run.phases[7])?.run.phases[7]?.data.threshold_used ?? 7
  const findings = keyFindings(metrics)
  const multi = series.length > 1

  // Steps that took under a second everywhere would be invisible bars; they're listed below the chart instead.
  const timedSteps = COLLECTION_PHASES.filter((id) => metrics.some((m) => (m.phaseMs[id] ?? 0) >= 1000))
  const quickSteps = COLLECTION_PHASES.filter(
    (id) => !timedSteps.includes(id) && metrics.some((m) => m.phaseMs[id] !== undefined),
  )
  const stepName = (id: PhaseNumber) => PHASES.find((p) => p.id === id)!.short

  // Likewise, steps that needed under 1 MB everywhere are listed instead of drawn.
  const hasMemory = metrics.some((m) => m.peakMemory !== undefined)
  const memorySteps = COLLECTION_PHASES.filter((id) => metrics.some((m) => (m.phaseMemory[id] ?? 0) >= 2 ** 20))
  const lightSteps = COLLECTION_PHASES.filter(
    (id) => !memorySteps.includes(id) && metrics.some((m) => m.phaseMemory[id] !== undefined),
  )

  const perSeries = (get: (m: CollectionMetrics) => number | undefined) =>
    Object.fromEntries(metrics.map((m) => [m.name, get(m)]))

  return (
    <div className="space-y-6">
      <SummaryCards metrics={metrics} series={series} />

      {findings.length > 0 && <KeyFindings findings={findings} />}

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard
          title="Which collection is fastest?"
          subtitle="Total time for steps 2–7. Shorter bar = faster. Query understanding runs once for all and isn't included."
        >
          <CollectionBarChart
            data={series.map((s) => ({ name: s.name, color: s.color, value: byName[s.name]?.totalMs }))}
            format={formatSeconds}
            yTitle="Time"
            valueName="Total time"
          />
        </ChartCard>

        <ChartCard
          title="Where does the time go?"
          subtitle="Time spent in each step, per collection."
          series={multi ? series : undefined}
          footnote={
            quickSteps.length > 0
              ? `${quickSteps.map(stepName).join(' and ')} took under a second everywhere, so ${quickSteps.length === 1 ? "it isn't" : "they aren't"} shown.`
              : undefined
          }
        >
          <GroupedBarChart
            categories={timedSteps.map((id) => ({ label: stepName(id), values: perSeries((m) => m.phaseMs[id]) }))}
            series={series}
            format={formatSeconds}
            xTitle="Time"
          />
        </ChartCard>

        <ChartCard
          title="Which collection needs the most memory?"
          subtitle="Extra memory the search server needed at its peak during steps 2–7, over what it held before. Shorter bar = lighter."
          footnote="Resident memory of the API process, sampled every 10 ms. Anything else running on the server at the same time is counted too."
        >
          {hasMemory ? (
            <CollectionBarChart
              data={series.map((s) => ({ name: s.name, color: s.color, value: toMB(byName[s.name]?.peakMemory) }))}
              format={formatMB}
              yTitle="Extra memory"
              valueName="Peak extra memory"
            />
          ) : (
            <EmptyChart>No memory measurements yet.</EmptyChart>
          )}
        </ChartCard>

        <ChartCard
          title="Where does the memory go?"
          subtitle="Extra memory each step needed at its peak, over what the server held when the step began."
          series={multi && hasMemory ? series : undefined}
          footnote={
            lightSteps.length > 0
              ? `${lightSteps.map(stepName).join(' and ')} needed under 1 MB everywhere, so ${lightSteps.length === 1 ? "it isn't" : "they aren't"} shown.`
              : undefined
          }
        >
          {memorySteps.length > 0 ? (
            <GroupedBarChart
              categories={memorySteps.map((id) => ({
                label: stepName(id),
                values: perSeries((m) => toMB(m.phaseMemory[id])),
              }))}
              series={series}
              format={formatMB}
              xTitle="Extra memory"
            />
          ) : (
            <EmptyChart>{hasMemory ? 'Every step needed under 1 MB.' : 'No memory measurements yet.'}</EmptyChart>
          )}
        </ChartCard>

        <ChartCard
          title="How many patents make it through each step?"
          subtitle={`Each step narrows the candidates down. The last group is the final results (score ≥ ${threshold}).`}
          series={multi ? series : undefined}
        >
          <GroupedBarChart
            categories={[
              { label: 'Retrieved', values: perSeries((m) => m.candidates) },
              { label: 'After filter', values: perSeries((m) => m.afterFilter) },
              { label: 'Verified', values: perSeries((m) => m.verified) },
              { label: 'Final results', values: perSeries((m) => m.results) },
            ]}
            series={series}
            format={formatCount}
            xTitle="Patents"
          />
        </ChartCard>

        <ChartCard
          title="How many chunks does each step handle?"
          subtitle="Chunk size changes how many pieces of text each step has to process."
          series={multi ? series : undefined}
        >
          <GroupedBarChart
            categories={[
              { label: 'Vector hits', values: perSeries((m) => m.chunkHits) },
              { label: 'Evidence', values: perSeries((m) => m.evidenceChunks) },
              { label: 'Reranked', values: perSeries((m) => m.chunksReranked) },
            ]}
            series={series}
            format={formatCount}
            xTitle="Chunks"
          />
        </ChartCard>

        <ChartCard
          title="How good are the results?"
          subtitle="Final score of the 1st, 2nd, 3rd… result. A higher line means better results; a longer line means more results."
          series={multi ? series : undefined}
        >
          <FinalScores metrics={metrics} series={series} threshold={threshold} />
        </ChartCard>

        <ChartCard
          title="How close are the best matches?"
          subtitle={`Vector similarity of the top ${TOP_CANDIDATES} candidates found in step 2. Higher = closer to the query.`}
          series={multi ? series : undefined}
        >
          <CandidateSimilarity metrics={metrics} series={series} />
        </ChartCard>
      </div>

      <Overlap metrics={metrics} series={series} />
      <CommonResults metrics={metrics} series={series} />
      <MetricsTable
        metrics={metrics}
        series={series}
        parseMs={Object.values(state.runs).find((r) => r.phases[1])?.phases[1]?.elapsedMs}
      />
      {/* Remounted when a collection runs on its own, so its tab opens. */}
      <DrillDown key={state.focus} state={state} series={series} />
    </div>
  )
}

function ChartCard({
  title,
  subtitle,
  series,
  footnote,
  children,
}: {
  title: string
  subtitle: string
  /** Shown as a legend above the chart. */
  series?: Series[]
  footnote?: string
  children: ReactNode
}) {
  return (
    <Card>
      <CardHeader title={title} subtitle={subtitle} />
      <CardBody className="space-y-3">
        {series && <Legend series={series} />}
        {children}
        {footnote && <p className="text-xs text-slate-500">{footnote}</p>}
      </CardBody>
    </Card>
  )
}

function KeyFindings({ findings }: { findings: Finding[] }) {
  return (
    <Card>
      <CardHeader title="Key findings" subtitle="Compared across the collections that finished." />
      <CardBody>
        <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {findings.map((f) => (
            <div key={f.label} className="rounded-lg bg-slate-50 px-4 py-3 ring-1 ring-slate-100">
              <dt className="text-xs font-medium tracking-wide text-slate-500 uppercase">{f.label}</dt>
              <dd className="mt-1 text-lg font-semibold text-slate-900">{f.value}</dd>
              <dd className="mt-0.5 text-xs text-slate-500">{f.detail}</dd>
            </div>
          ))}
        </dl>
      </CardBody>
    </Card>
  )
}

function SummaryCards({ metrics, series }: { metrics: CollectionMetrics[]; series: Series[] }) {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {metrics.map((m, i) => (
        <Card key={m.name} className="p-4">
          <div className="mb-3 flex items-center gap-2 font-semibold text-slate-900">
            <span className="size-3 rounded-sm" style={{ backgroundColor: series[i].color }} aria-hidden />
            {m.name}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Time" value={orDash(m.totalMs, formatMs)} hint="Steps 2–7" />
            <Stat
              label="Results"
              value={orDash(m.results, formatCount)}
              hint={orDash(m.rejected, (r) => `${r} below threshold`)}
            />
            <Stat
              label="Top score"
              value={orDash(m.topScore, formatScore2)}
              hint={orDash(m.meanScore, (s) => `mean ${s.toFixed(2)}`)}
            />
            <Stat
              label="Candidates"
              value={orDash(m.candidates, formatCount)}
              hint={orDash(m.chunkHits, (h) => `${formatCount(h)} chunk hits`)}
            />
          </div>
        </Card>
      ))}
    </div>
  )
}

function EmptyChart({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-[280px] items-center justify-center rounded-lg bg-slate-50 text-sm text-slate-500">
      {children}
    </div>
  )
}

function FinalScores({
  metrics,
  series,
  threshold,
}: {
  metrics: CollectionMetrics[]
  series: Series[]
  threshold: number
}) {
  const scores = metrics.flatMap((m) => m.finalScores.map((s) => s.score))
  if (scores.length === 0) return <EmptyChart>No qualifying results yet.</EmptyChart>
  const byName = Object.fromEntries(metrics.map((m) => [m.name, m]))
  // Start one point below the threshold (or lowest score) so the drop-off is visible.
  const lo = Math.max(0, Math.min(threshold, ...scores) - 1)
  return (
    <RankLineChart
      series={series}
      values={(name) => byName[name]?.finalScores.map((s) => s.score) ?? []}
      format={(v) => (Number.isInteger(v) ? String(v) : v.toFixed(2))}
      yTitle="Final score"
      yRange={[lo, 10]}
      reference={{ value: threshold, label: `Threshold ${threshold}` }}
    />
  )
}

function CandidateSimilarity({ metrics, series }: { metrics: CollectionMetrics[]; series: Series[] }) {
  const scores = metrics.flatMap((m) => m.topSimilarities.map((s) => s.score))
  if (scores.length === 0) return <EmptyChart>No candidates yet.</EmptyChart>
  const byName = Object.fromEntries(metrics.map((m) => [m.name, m]))
  // Top candidates sit close together, so the axis zooms in on their range rather than start at 0.
  return (
    <RankLineChart
      series={series}
      values={(name) => byName[name]?.topSimilarities.map((s) => s.score) ?? []}
      format={(v) => v.toFixed(2)}
      yTitle="Similarity"
      yRange={[Math.min(...scores), Math.max(...scores)]}
    />
  )
}

function Overlap({ metrics, series }: { metrics: CollectionMetrics[]; series: Series[] }) {
  const rows = resultOverlap(metrics)
  const pairs = metrics.flatMap((a, i) => metrics.slice(i + 1).map((b) => [a, b] as const))

  const columns: Column<OverlapRow>[] = [
    {
      header: 'Patent',
      render: (r) => (
        <div className="min-w-48">
          <div className="font-mono text-xs text-slate-900">{r.patentId}</div>
          {r.title && <div className="line-clamp-1 text-xs text-slate-500">{r.title}</div>}
        </div>
      ),
    },
    ...metrics.map<Column<OverlapRow>>((m, i) => ({
      header: m.name,
      align: 'right',
      render: (r) => {
        const f = r.found[m.name]
        return f ? (
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full" style={{ backgroundColor: series[i].color }} aria-hidden />#{f.rank}
            {f.score !== null && <> · {f.score.toFixed(2)}</>}
          </span>
        ) : (
          <span className="text-slate-400">—</span>
        )
      },
    })),
    { header: 'Found in', align: 'right', render: (r) => `${Object.keys(r.found).length} / ${metrics.length}` },
  ]

  return (
    <Card>
      <CardHeader
        title="Result overlap"
        subtitle="Which qualifying patents each collection found, with rank and final score."
      />
      <CardBody className="space-y-4">
        {pairs.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {pairs.map(([a, b]) => {
              const sharedResults = rows.filter((r) => r.found[a.name] && r.found[b.name]).length
              const shared = sharedCandidates(a, b)
              return (
                <div
                  key={`${a.name}-${b.name}`}
                  className="rounded-lg bg-slate-50 px-3 py-2 text-sm ring-1 ring-slate-100"
                >
                  <div className="font-medium text-slate-700">
                    {a.name} ∩ {b.name}
                  </div>
                  <div className="text-slate-500">
                    {sharedResults} shared result{sharedResults === 1 ? '' : 's'}
                    {shared !== undefined && ` · ${shared} shared candidates`}
                  </div>
                </div>
              )
            })}
          </div>
        )}
        <DataTable
          columns={columns}
          rows={rows}
          rowKey={(r) => r.patentId}
          emptyText="No collection has qualifying results yet."
        />
      </CardBody>
    </Card>
  )
}

/** Patents that every selected collection found among its qualifying results. */
function CommonResults({ metrics, series }: { metrics: CollectionMetrics[]; series: Series[] }) {
  const rows = resultOverlap(metrics).filter((r) => Object.keys(r.found).length === metrics.length)

  const columns: Column<OverlapRow>[] = [
    {
      header: 'Patent',
      render: (r) => (
        <div className="min-w-48">
          <div className="font-mono text-xs text-slate-900">{r.patentId}</div>
          {r.title && <div className="line-clamp-1 text-xs text-slate-500">{r.title}</div>}
        </div>
      ),
    },
    ...metrics.map<Column<OverlapRow>>((m, i) => ({
      header: m.name,
      align: 'right',
      render: (r) => {
        const f = r.found[m.name]
        return f ? (
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full" style={{ backgroundColor: series[i].color }} aria-hidden />#{f.rank}
            {f.score !== null && <> · {f.score.toFixed(2)}</>}
          </span>
        ) : (
          <span className="text-slate-400">—</span>
        )
      },
    })),
  ]

  return (
    <Card>
      <CardHeader
        title="Common results"
        subtitle={
          metrics.length < 2
            ? 'Run more than one collection to see patents common to all of them.'
            : `Qualifying patents found by all ${metrics.length} collections, with rank and final score in each.`
        }
      />
      <CardBody>
        <DataTable
          columns={columns}
          rows={rows}
          rowKey={(r) => r.patentId}
          emptyText={metrics.length < 2 ? 'Run more than one collection to compare.' : 'No patent was found by every collection.'}
        />
      </CardBody>
    </Card>
  )
}

interface MetricRow {
  /** Defaults to the label. */
  key?: string
  label: string
  value: (m: CollectionMetrics) => string
}

function MetricsTable({
  metrics,
  series,
  parseMs,
}: {
  metrics: CollectionMetrics[]
  series: Series[]
  parseMs?: number
}) {
  const rows: MetricRow[] = [
    { label: 'Chunk size (max tokens)', value: (m) => orDash(m.chunkTokens, formatCount) },
    { label: 'Total patents in collection', value: (m) => orDash(m.collectionPatents, formatCount) },
    { label: 'Total chunks / vectors in collection', value: (m) => orDash(m.collectionChunks, formatCount) },
    { label: 'Total time (steps 2–7)', value: (m) => orDash(m.totalMs, formatMs) },
    ...COLLECTION_PHASES.map<MetricRow>((id) => ({
      label: `  ${PHASES.find((p) => p.id === id)!.title}`,
      value: (m) => orDash(m.phaseMs[id], formatMs),
    })),
    { key: 'embedding-ms', label: '  Embedding (within retrieval)', value: (m) => orDash(m.embeddingMs, formatMs) },
    { key: 'qdrant-ms', label: '  Qdrant search (within retrieval)', value: (m) => orDash(m.qdrantMs, formatMs) },
    { label: 'Peak extra memory (steps 2–7)', value: (m) => orDash(m.peakMemory, formatBytes) },
    ...COLLECTION_PHASES.map<MetricRow>((id) => ({
      key: `memory-${id}`,
      label: `  ${PHASES.find((p) => p.id === id)!.title}`,
      value: (m) => orDash(m.phaseMemory[id], formatBytes),
    })),
    { label: 'Peak server memory', value: (m) => orDash(m.peakRss, formatBytes) },
    { label: 'Candidates retrieved', value: (m) => orDash(m.candidates, formatCount) },
    { label: 'After metadata filter', value: (m) => orDash(m.afterFilter, formatCount) },
    { label: 'Verified', value: (m) => orDash(m.verified, formatCount) },
    { label: 'Qualifying results', value: (m) => orDash(m.results, formatCount) },
    { label: 'Below threshold', value: (m) => orDash(m.rejected, formatCount) },
    { label: 'Top final score', value: (m) => orDash(m.topScore, formatScore2) },
    { label: 'Mean final score', value: (m) => orDash(m.meanScore, formatScore2) },
    { label: 'Top-1 reranker score', value: (m) => orDash(m.topRerankerScore, (s) => s.toFixed(3)) },
    { label: 'Mean reranker score', value: (m) => orDash(m.meanRerankerScore, (s) => s.toFixed(3)) },
    { label: 'Min reranker score', value: (m) => orDash(m.minRerankerScore, (s) => s.toFixed(3)) },
    { label: 'Best candidate similarity', value: (m) => orDash(m.topSimilarities[0]?.score, (s) => s.toFixed(3)) },
    { label: 'Vector chunk hits', value: (m) => orDash(m.chunkHits, formatCount) },
    { label: 'Evidence chunks', value: (m) => orDash(m.evidenceChunks, formatCount) },
    { label: 'Chunks reranked', value: (m) => orDash(m.chunksReranked, formatCount) },
  ]

  const columns: Column<MetricRow>[] = [
    { header: 'Metric', render: (r) => <span className="whitespace-pre text-slate-600">{r.label}</span> },
    ...metrics.map<Column<MetricRow>>((m, i) => ({
      header: m.name,
      align: 'right',
      render: (r) => (
        <span className="inline-flex items-center gap-1.5">
          {r === rows[0] && (
            <span className="size-2 rounded-full" style={{ backgroundColor: series[i].color }} aria-hidden />
          )}
          {r.value(m)}
        </span>
      ),
    })),
  ]

  return (
    <Card>
      <CardHeader
        title="All numbers"
        subtitle={`Every value behind the charts. Query understanding ran once for all collections${parseMs === undefined ? '' : ` (${formatMs(parseMs)})`}.`}
      />
      <CardBody>
        <DataTable columns={columns} rows={rows} rowKey={(r) => r.key ?? r.label} />
      </CardBody>
    </Card>
  )
}

/** The regular results and pipeline views for one collection's run. */
function DrillDown({ state, series }: { state: CompareState; series: Series[] }) {
  const [chosen, setChosen] = useState<string | null>(null)
  const active = chosen && state.runs[chosen] ? chosen : (state.focus ?? state.collections[0])
  const run = active ? state.runs[active] : undefined
  if (!run) return null

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Details per collection</h2>
        <Tabs
          value={active}
          onChange={setChosen}
          items={series.map((s) => ({
            id: s.name,
            label: s.name,
            count: state.runs[s.name]?.phases[7]?.data.results.length,
          }))}
        />
      </div>
      <SearchRunView key={active} run={run} />
    </section>
  )
}

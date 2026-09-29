/**
 * Comparison charts (Recharts). One colour per collection from the --series-N
 * tokens (index.css); axes and grid use the theme's slate tokens, so both follow
 * light and dark mode. Every value is also in the "All numbers" table.
 */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export interface Series {
  name: string
  color: string
}

const TICK = { fill: 'var(--color-slate-500)', fontSize: 12 }
const AXIS_LINE = 'var(--color-slate-300)'
const GRID = 'var(--color-slate-200)'
const VALUE_LABEL = { fill: 'var(--color-slate-600)', fontSize: 11 }

/** Round tick values (steps of 1, 2, 2.5 or 5 × 10ⁿ) covering [min, max]. */
function niceTicks(min: number, max: number, count = 5): number[] {
  if (max <= min) max = min + 1
  const raw = (max - min) / (count - 1)
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const n = raw / magnitude
  const step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * magnitude
  const ticks: number[] = []
  for (let v = Math.floor(min / step) * step; v <= Math.ceil(max / step) * step + step / 2; v += step) {
    ticks.push(Number(v.toFixed(10)))
  }
  return ticks
}

function axisTitle(value: string, vertical: boolean) {
  return vertical
    ? { value, angle: -90, position: 'insideLeft' as const, style: { textAnchor: 'middle' as const }, ...TICK }
    : { value, position: 'insideBottom' as const, offset: -12, ...TICK }
}

interface LabelProps {
  x?: number | string
  y?: number | string
  width?: number | string
  height?: number | string
  value?: unknown
}

/** Value label at a bar's tip, on one line (Recharts would wrap it to the bar's width). */
function barLabel(format: (value: number) => string, side: 'top' | 'right') {
  return function BarLabel({ x = 0, y = 0, width = 0, height = 0, value }: LabelProps) {
    if (typeof value !== 'number') return null
    const [bx, by, bw, bh] = [x, y, width, height].map(Number)
    return side === 'top' ? (
      <text x={bx + bw / 2} y={by - 6} textAnchor="middle" {...VALUE_LABEL}>
        {format(value)}
      </text>
    ) : (
      <text x={bx + bw + 6} y={by + bh / 2} dominantBaseline="central" {...VALUE_LABEL}>
        {format(value)}
      </text>
    )
  }
}

/** Swatch + name per series, shown above a chart with more than one series. */
export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-slate-700">
      {series.map((s) => (
        <li key={s.name} className="flex items-center gap-2">
          <span className="size-3 rounded-sm" style={{ backgroundColor: s.color }} aria-hidden />
          {s.name}
        </li>
      ))}
    </ul>
  )
}

interface TooltipRow {
  name?: string | number
  value?: unknown
  color?: string
  payload?: { color?: string }
}

/** Hover readout: the category, then each series' value (value first, name second). */
function ChartTooltip({
  active,
  payload,
  label,
  format,
  title,
}: {
  active?: boolean
  payload?: readonly TooltipRow[]
  label?: string | number
  format: (value: number) => string
  title: (label: string | number | undefined) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-md bg-surface px-3 py-2 text-xs shadow-lg ring-1 ring-slate-200">
      <div className="mb-1 font-medium text-slate-500">{title(label)}</div>
      {payload.map((p) => (
        <div key={String(p.name)} className="flex items-center gap-2 py-0.5">
          <span
            className="h-0.5 w-3 rounded-full"
            style={{ backgroundColor: p.payload?.color ?? p.color }}
            aria-hidden
          />
          <span className="font-semibold text-slate-900 tabular-nums">
            {typeof p.value === 'number' ? format(p.value) : '—'}
          </span>
          <span className="text-slate-500">{p.name}</span>
        </div>
      ))}
    </div>
  )
}

interface CollectionBarChartProps {
  data: { name: string; color: string; value?: number }[]
  format: (value: number) => string
  yTitle: string
  valueName: string
}

/** One column per collection, value on top. */
export function CollectionBarChart({ data, format, yTitle, valueName }: CollectionBarChartProps) {
  const ticks = niceTicks(0, Math.max(0, ...data.map((d) => d.value ?? 0)))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 24, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="name" tick={TICK} tickLine={false} stroke={AXIS_LINE} />
        <YAxis
          ticks={ticks}
          domain={[0, ticks.at(-1)!]}
          tick={TICK}
          tickLine={false}
          axisLine={false}
          tickFormatter={format}
          label={axisTitle(yTitle, true)}
          width={72}
        />
        <Tooltip
          cursor={{ fill: 'var(--color-slate-100)' }}
          content={(props) => <ChartTooltip {...props} format={format} title={(label) => `Collection ${label}`} />}
        />
        <Bar dataKey="value" name={valueName} maxBarSize={48} radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.name} fill={d.color} />
          ))}
          <LabelList dataKey="value" content={barLabel(format, 'top')} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

interface GroupedBarChartProps {
  /** One group per category; values keyed by series name. */
  categories: { label: string; values: Record<string, number | undefined> }[]
  series: Series[]
  format: (value: number) => string
  xTitle: string
}

const BAR_SIZE = 14

/**
 * Horizontal bars: one group per category (down the side), one bar per
 * collection, value at the bar's tip — so labels never collide.
 */
export function GroupedBarChart({ categories, series, format, xTitle }: GroupedBarChartProps) {
  const data = categories.map((c) => ({ label: c.label, ...c.values }))
  const max = Math.max(0, ...categories.flatMap((c) => series.map((s) => c.values[s.name] ?? 0)))
  const ticks = niceTicks(0, max)
  // Bars 2px apart within a group, 16px between groups, plus room for the axis.
  const height = categories.length * (series.length * (BAR_SIZE + 2) + 16) + 56
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout="vertical"
        barGap={2}
        barCategoryGap={8}
        margin={{ top: 0, right: 56, bottom: 16, left: 0 }}
      >
        <CartesianGrid horizontal={false} stroke={GRID} />
        <XAxis
          type="number"
          ticks={ticks}
          domain={[0, ticks.at(-1)!]}
          tick={TICK}
          tickLine={false}
          axisLine={false}
          tickFormatter={format}
          label={axisTitle(xTitle, false)}
        />
        <YAxis type="category" dataKey="label" tick={TICK} tickLine={false} stroke={AXIS_LINE} width={96} />
        <Tooltip
          cursor={{ fill: 'var(--color-slate-100)' }}
          content={(props) => <ChartTooltip {...props} format={format} title={(label) => String(label)} />}
        />
        {series.map((s) => (
          <Bar
            key={s.name}
            dataKey={s.name}
            name={s.name}
            fill={s.color}
            barSize={BAR_SIZE}
            radius={[0, 4, 4, 0]}
            isAnimationActive={false}
          >
            <LabelList dataKey={s.name} content={barLabel(format, 'right')} />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

interface RankLineChartProps {
  series: Series[]
  /** Values in rank order (best first), per series name. */
  values: (series: string) => number[]
  format: (value: number) => string
  yTitle: string
  /** Lowest and highest value the y-axis must show; it's widened to round ticks. */
  yRange: [number, number]
  /** A horizontal reference line, e.g. the score threshold. */
  reference?: { value: number; label: string }
}

/** Value by rank (1 = best), one line per collection: shows how quickly quality drops off. */
export function RankLineChart({ series, values, format, yTitle, yRange, reference }: RankLineChartProps) {
  const byName = Object.fromEntries(series.map((s) => [s.name, values(s.name)]))
  const length = Math.max(0, ...Object.values(byName).map((v) => v.length))
  const data = Array.from({ length }, (_, i) => ({
    rank: i + 1,
    ...Object.fromEntries(series.map((s) => [s.name, byName[s.name][i]])),
  }))
  const ticks = niceTicks(yRange[0], yRange[1])
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 16, right: 16, bottom: 16, left: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis
          dataKey="rank"
          tick={TICK}
          tickLine={false}
          stroke={AXIS_LINE}
          label={axisTitle('Rank (1 = best)', false)}
        />
        <YAxis
          ticks={ticks}
          domain={[ticks[0], ticks.at(-1)!]}
          tick={TICK}
          tickLine={false}
          axisLine={false}
          tickFormatter={format}
          label={axisTitle(yTitle, true)}
          width={72}
        />
        <Tooltip
          cursor={{ stroke: 'var(--color-slate-300)' }}
          content={(props) => <ChartTooltip {...props} format={format} title={(label) => `Rank ${label}`} />}
        />
        {reference && (
          <ReferenceLine
            y={reference.value}
            stroke="var(--color-slate-400)"
            label={{ value: reference.label, position: 'insideTopRight', ...VALUE_LABEL }}
          />
        )}
        {series.map((s) => (
          <Line
            key={s.name}
            dataKey={s.name}
            name={s.name}
            stroke={s.color}
            strokeWidth={2}
            dot={{ r: 4, fill: s.color, stroke: 'var(--color-surface)', strokeWidth: 2 }}
            activeDot={{ r: 6, fill: s.color, stroke: 'var(--color-surface)', strokeWidth: 2 }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

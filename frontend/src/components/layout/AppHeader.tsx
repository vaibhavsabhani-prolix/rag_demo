import clsx from 'clsx'
import { Link, NavLink } from 'react-router'
import { Button, ChartIcon, DatabaseIcon, HistoryIcon, SearchIcon, SettingsIcon, SparklesIcon } from '@/components/ui'
import { useAppDispatch } from '@/store'
import { settingsToggled } from '@/store/uiSlice'

const NAV = [
  { to: '/', label: 'Search', icon: SearchIcon, end: true },
  { to: '/features', label: 'Feature Search', icon: SparklesIcon, end: false },
  { to: '/compare', label: 'Compare', icon: ChartIcon, end: false },
  { to: '/collections', label: 'Collections', icon: DatabaseIcon, end: false },
  { to: '/history', label: 'History', icon: HistoryIcon, end: false },
]

export function AppHeader() {
  const dispatch = useAppDispatch()
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-surface/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between gap-4 px-4">
        <div className="flex items-center gap-6">
          <Link to="/" className="flex items-center gap-2.5">
            <img src="/favicon.svg" alt="" className="size-7" />
            <span className="hidden font-semibold text-slate-900 sm:inline">Patent Search</span>
          </Link>
          <nav className="flex gap-1">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                aria-label={label}
                title={label}
                className={({ isActive }) =>
                  clsx(
                    'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                    isActive ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
                  )
                }
              >
                <Icon className="size-4" />
                {/* Icons only on phones, so the header fits. */}
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
        <Button variant="ghost" size="sm" onClick={() => dispatch(settingsToggled(true))}>
          <SettingsIcon className="size-4" />
          <span className="hidden sm:inline">Settings</span>
        </Button>
      </div>
    </header>
  )
}

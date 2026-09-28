import { Outlet } from 'react-router'
import { ApplyAppearance } from '@/features/settings/ApplyAppearance'
import { SettingsDrawer } from '@/features/settings/SettingsDrawer'
import { AppHeader } from './AppHeader'

/** Shared page frame: header, page content, and the settings drawer. */
export function AppLayout() {
  return (
    <div className="min-h-screen">
      <ApplyAppearance />
      <AppHeader />
      <main className="mx-auto max-w-5xl px-4 py-10">
        <Outlet />
      </main>
      <SettingsDrawer />
    </div>
  )
}

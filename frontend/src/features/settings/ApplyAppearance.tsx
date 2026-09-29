import { useEffect, useState } from 'react'
import { useUiSettings } from '@/hooks/queries'
import { DEFAULT_UI_SETTINGS, type Theme } from '@/schemas/settings'

/** localStorage key read by the inline script in index.html to set the theme before first paint. */
export const THEME_STORAGE_KEY = 'theme'

const darkQuery = window.matchMedia('(prefers-color-scheme: dark)')

/** Follows the OS light/dark preference, updating when it changes. */
function useSystemPrefersDark(): boolean {
  const [dark, setDark] = useState(darkQuery.matches)
  useEffect(() => {
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches)
    darkQuery.addEventListener('change', onChange)
    return () => darkQuery.removeEventListener('change', onChange)
  }, [])
  return dark
}

/**
 * Applies the saved appearance settings: the theme (the "dark" class on <html>)
 * and the highlight colour (the --highlight CSS variable, see index.css).
 */
export function ApplyAppearance() {
  const { data } = useUiSettings()
  const systemDark = useSystemPrefersDark()

  const theme: Theme = data?.theme ?? 'system'
  const dark = theme === 'dark' || (theme === 'system' && systemDark)
  const highlightColor = data?.highlight_color ?? DEFAULT_UI_SETTINGS.highlight_color

  useEffect(() => {
    document.documentElement.style.setProperty('--highlight', highlightColor)
  }, [highlightColor])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      // storage unavailable (private mode); the theme still applies for this visit
    }
  }, [dark, theme])

  return null
}

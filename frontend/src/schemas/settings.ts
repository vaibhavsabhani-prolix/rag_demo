/** Appearance settings: theme and highlight colour (app/api/schemas.py → UiSettings). */
import { z } from 'zod'

export const THEMES = ['system', 'light', 'dark'] as const
export type Theme = (typeof THEMES)[number]

export const uiSettingsSchema = z.object({
  theme: z.enum(THEMES),
  highlight_color: z.string().regex(/^#[0-9a-f]{6}$/i, 'Use a colour like #f59e0b'),
})

export type UiSettings = z.infer<typeof uiSettingsSchema>

export const DEFAULT_UI_SETTINGS: UiSettings = { theme: 'system', highlight_color: '#f59e0b' }

/** Appearance settings: theme (app/api/schemas.py → UiSettings). */
import { z } from 'zod'

export const THEMES = ['system', 'light', 'dark'] as const
export type Theme = (typeof THEMES)[number]

export const uiSettingsSchema = z.object({
  theme: z.enum(THEMES),
})

export type UiSettings = z.infer<typeof uiSettingsSchema>

export const DEFAULT_UI_SETTINGS: UiSettings = { theme: 'system' }

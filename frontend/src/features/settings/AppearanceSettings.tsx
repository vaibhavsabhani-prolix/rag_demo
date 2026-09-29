import { zodResolver } from '@hookform/resolvers/zod'
import { Controller, useForm } from 'react-hook-form'
import type { CSSProperties } from 'react'
import { Alert, Button, ColorField, Section, SegmentedControl } from '@/components/ui'
import { useSaveUiSettings, useUiSettings } from '@/hooks/queries'
import { DEFAULT_UI_SETTINGS, uiSettingsSchema, type Theme, type UiSettings } from '@/schemas/settings'

const THEME_OPTIONS: { value: Theme; label: string }[] = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
]

const HIGHLIGHT_PRESETS = [
  { name: 'Amber', value: '#f59e0b' },
  { name: 'Yellow', value: '#eab308' },
  { name: 'Green', value: '#22c55e' },
  { name: 'Sky', value: '#0ea5e9' },
  { name: 'Violet', value: '#8b5cf6' },
  { name: 'Pink', value: '#ec4899' },
]

/** A sample sentence showing how highlights look in `color`, before saving. */
function HighlightPreview({ color }: { color?: string }) {
  return (
    <p
      className="rounded-lg border border-slate-200 bg-slate-50/60 p-3 text-sm leading-relaxed text-slate-700"
      style={color ? ({ '--highlight': color } as CSSProperties) : undefined}
    >
      <mark className="hl-sentence hl-sentence-strong">
        The apparatus condenses atmospheric moisture into liquid <span className="hl-term">water</span>.
      </mark>{' '}
      <mark className="hl-sentence">
        A fan draws ambient <span className="hl-term">air</span> through the cooled fins.
      </mark>{' '}
      The housing is made of aluminium.
    </p>
  )
}

/** Theme and highlight colour; saved to the server database. */
export function AppearanceSettings() {
  const { data } = useUiSettings()
  const save = useSaveUiSettings()
  const {
    control,
    handleSubmit,
    reset,
    formState: { isDirty },
  } = useForm<UiSettings>({
    resolver: zodResolver(uiSettingsSchema),
    values: data, // stays in sync with the saved settings
  })

  const onSubmit = handleSubmit((values) => save.mutate(values))

  return (
    <Section title="Appearance">
      <form onSubmit={onSubmit} className="space-y-5">
        <Controller
          control={control}
          name="theme"
          render={({ field }) => (
            <SegmentedControl
              label="Theme"
              options={THEME_OPTIONS}
              value={field.value ?? DEFAULT_UI_SETTINGS.theme}
              onChange={field.onChange}
            />
          )}
        />

        <Controller
          control={control}
          name="highlight_color"
          render={({ field, fieldState }) => (
            <div className="space-y-2">
              <ColorField
                label="Highlight colour"
                value={field.value ?? DEFAULT_UI_SETTINGS.highlight_color}
                onChange={field.onChange}
                presets={HIGHLIGHT_PRESETS}
                error={fieldState.error?.message}
              />
              <HighlightPreview color={field.value} />
            </div>
          )}
        />

        {save.error && <Alert tone="danger">{save.error.message}</Alert>}

        <div className="flex items-center justify-between gap-2">
          <Button variant="ghost" size="sm" onClick={() => reset(DEFAULT_UI_SETTINGS, { keepDefaultValues: true })}>
            Reset to default
          </Button>
          <div className="flex items-center gap-3">
            {save.isSuccess && !isDirty && <span className="text-sm text-emerald-600">Saved</span>}
            <Button type="submit" size="sm" disabled={!isDirty} loading={save.isPending}>
              Save
            </Button>
          </div>
        </div>
      </form>
    </Section>
  )
}

import { zodResolver } from '@hookform/resolvers/zod'
import { Controller, useForm } from 'react-hook-form'
import { Alert, Button, Section, SegmentedControl } from '@/components/ui'
import { useSaveUiSettings, useUiSettings } from '@/hooks/queries'
import { DEFAULT_UI_SETTINGS, uiSettingsSchema, type Theme, type UiSettings } from '@/schemas/settings'

const THEME_OPTIONS: { value: Theme; label: string }[] = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
]

/** Theme; saved to the server database. */
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

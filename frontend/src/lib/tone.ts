/** Maps pipeline values to the shared colour tones used by Badge / ScoreBar. */
export type Tone = 'neutral' | 'brand' | 'success' | 'warning' | 'danger' | 'info'

/** null for a metadata-only query's results, which aren't scored. */
export function finalScoreTone(score: number | null): Tone {
  if (score === null) return 'neutral'
  if (score >= 8.5) return 'success'
  if (score >= 7.5) return 'info'
  return 'brand'
}

export function unitScoreTone(score: number): Tone {
  if (score >= 0.8) return 'success'
  if (score >= 0.5) return 'warning'
  return 'danger'
}

export function verificationTone(status: string): Tone {
  switch (status) {
    case 'SUPPORTED':
      return 'success'
    case 'CONTRADICTED':
      return 'danger'
    case 'NOT_SUPPORTED':
      return 'warning'
    default:
      return 'neutral'
  }
}

// Display names and copy for the model's answers (Phase 2 model: healthy / rust / other, plus "not sure").
// Worded so that no answer reads as a diagnosis.
import type { Prediction } from './api'

export type Outcome = 'rust' | 'other' | 'healthy' | 'not_sure'

export const LABELS: Record<string, string> = {
  healthy: 'Looks healthy',
  rust: 'Rust-like symptoms',
  other: 'Another leaf problem',
  not_sure: 'Not sure',
}

export const displayName = (id: string) => LABELS[id] ?? id

type OutcomeCopy = {
  title: string
  /** One-line summary used on the home page. */
  short: string
  /** What the answer means, shown on the result. */
  explain: string
  /** Practical next steps, shown on the result. */
  next: string[]
}

export const OUTCOMES: Record<Outcome, OutcomeCopy> = {
  rust: {
    title: 'Rust-like symptoms',
    short: 'Signs that look like coffee leaf rust, the most damaging coffee leaf disease.',
    explain:
      'The leaf shows signs that look like coffee leaf rust. Some other problems, such as mite damage, can look similar.',
    next: [
      'Turn the leaf over: rust usually shows as yellow-orange powder on the underside.',
      'Check other leaves and nearby plants to see how far it may have spread.',
      'Ask an agronomist or extension officer to confirm before treating.',
    ],
  },
  other: {
    title: 'Another leaf problem',
    short: 'A problem that is not rust, such as leaf spots or leaf miner damage.',
    explain:
      'The leaf shows a problem that does not look like rust, for example leaf spots (phoma, cercospora) or leaf ' +
      'miner damage. The app does not say which one.',
    next: [
      'Note which leaves are affected and what the marks look like.',
      'Ask an agronomist or extension officer to identify the problem.',
      'Check a few more leaves to see whether the answer stays the same.',
    ],
  },
  healthy: {
    title: 'Looks healthy',
    short: 'No signs of rust or of the other leaf diseases the model knows.',
    explain:
      'No signs of rust or other common leaf diseases were found on this leaf. Other leaves on the plant may differ.',
    next: [
      'Check a few more leaves: one leaf does not show the whole plant.',
      'Keep checking regularly, especially during the rainy season when rust spreads fastest.',
    ],
  },
  not_sure: {
    title: 'Not sure',
    short: 'The photo is unclear or unlike anything it learned from, so it declines to guess.',
    explain: '',
    next: [],
  },
}

/** The answer the app shows: the model's label, or "not sure" when the backend declined to answer. */
export function outcomeOf(result: Prediction): Outcome {
  if (result.uncertain) return 'not_sure'
  return (['rust', 'other', 'healthy'] as const).find((o) => o === result.label) ?? 'not_sure'
}

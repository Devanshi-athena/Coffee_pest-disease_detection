import type { Prediction } from '../api'
import { displayName } from '../labels'

const pct = (x: number) => `${Math.round(x * 100)}%`

// What each answer means for the user. Worded so that no answer reads as a diagnosis.
const EXPLAIN: Record<string, string> = {
  rust:
    'The leaf shows signs that look like coffee leaf rust. Other problems, such as mite damage, can look similar.',
  healthy:
    'No signs of rust or other common leaf diseases were found on this leaf. Other leaves on the plant may differ.',
  other:
    'The leaf shows a problem that does not look like rust, for example leaf spots or leaf miner damage. ' +
    'This app cannot tell which one.',
}

export function ResultCard({ result }: { result: Prediction }) {
  if (result.uncertain) {
    const unfamiliar = result.reason === 'unfamiliar_image'
    return (
      <section className="card card--warn" aria-live="polite">
        <h2>Not sure</h2>
        <p>
          {unfamiliar
            ? "This photo doesn't look like the coffee leaf photos the model learned from, so it won't guess."
            : "The model can't tell clearly from this photo."}{' '}
          Try another photo with:
        </p>
        <ul className="tips">
          <li>One coffee leaf, filling most of the frame</li>
          <li>Daylight, no harsh shadows or flash glare</li>
          <li>Sharp focus, not blurry</li>
        </ul>
        {/* An unfamiliar photo gets no guesses at all: any guess could mislead. */}
        {!unfamiliar && (
          <>
            <p className="eyebrow">Possible, but not reliable</p>
            <TopGuesses result={result} muted />
          </>
        )}
      </section>
    )
  }

  return (
    <section className={`card${result.label === 'rust' ? ' card--alert' : ''}`} aria-live="polite">
      <p className="eyebrow">Result</p>
      <h2>{displayName(result.label)}</h2>
      <p>{EXPLAIN[result.label] ?? ''}</p>
      <ConfidenceBar value={result.confidence} />
      <TopGuesses result={result} />
      <p className="note">
        This is a screening estimate, not a diagnosis. Confirm with an agronomist or extension officer before
        treating.
      </p>
    </section>
  )
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="conf">
      <div
        className="conf__track"
        role="progressbar"
        aria-label="Confidence"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
      >
        <div className="conf__fill" style={{ width: pct(value) }} />
      </div>
      <span className="conf__value">{pct(value)} confidence</span>
    </div>
  )
}

function TopGuesses({ result, muted = false }: { result: Prediction; muted?: boolean }) {
  return (
    <ol className={`top3${muted ? ' top3--muted' : ''}`} aria-label="All answers by confidence">
      {result.top3.map((g) => (
        <li key={g.label}>
          <span>{displayName(g.label)}</span>
          <span className="top3__bar">
            <span style={{ width: pct(g.confidence) }} />
          </span>
          <span className="top3__pct">{pct(g.confidence)}</span>
        </li>
      ))}
    </ol>
  )
}

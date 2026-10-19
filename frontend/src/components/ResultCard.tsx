import type { Prediction } from '../api'
import { displayName, OUTCOMES, outcomeOf, type Outcome } from '../labels'
import { CheckIcon, FocusIcon, LeafIcon, OutcomeIcon, SunIcon } from './Icons'

const pct = (x: number) => `${Math.round(x * 100)}%`

export function ResultCard({ result }: { result: Prediction }) {
  const outcome = outcomeOf(result)
  const copy = OUTCOMES[outcome]

  return (
    <section className={`result result--${outcome}`} aria-live="polite" aria-labelledby="result-title">
      <header className="result__head">
        <span className="result__badge">
          <OutcomeIcon outcome={outcome} />
        </span>
        <div>
          <p className="eyebrow">Result</p>
          <h2 id="result-title" className="result__title">{copy.title}</h2>
        </div>
      </header>

      {outcome === 'not_sure' ? <NotSure result={result} /> : <Answered result={result} outcome={outcome} />}

      <p className="result__note">
        A screening estimate, not a diagnosis. Confirm with an agronomist or extension officer before treating.
      </p>
    </section>
  )
}

function Answered({ result, outcome }: { result: Prediction; outcome: Outcome }) {
  const copy = OUTCOMES[outcome]
  return (
    <>
      <p className="result__explain">{copy.explain}</p>
      <ConfidenceMeter value={result.confidence} threshold={result.threshold} />
      <details className="breakdown">
        <summary>See all answers</summary>
        <Breakdown result={result} />
      </details>
      <div className="next">
        <h3>What to do next</h3>
        <ul className="checklist">
          {copy.next.map((n) => (
            <li key={n}>
              <CheckIcon />
              <span>{n}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  )
}

function NotSure({ result }: { result: Prediction }) {
  const unfamiliar = result.reason === 'unfamiliar_image'
  return (
    <>
      <p className="result__explain">
        {unfamiliar
          ? "This photo doesn't look like the coffee leaf photos the model learned from, so it won't guess. It may show a problem the model does not know, or the photo may be framed differently."
          : "The model can't tell clearly from this photo, so it won't guess."}
      </p>
      <div className="next">
        <h3>Try another photo with</h3>
        <ul className="tips-row">
          <li><LeafIcon /><span>One leaf filling most of the frame</span></li>
          <li><SunIcon /><span>Daylight, no harsh shadows or flash</span></li>
          <li><FocusIcon /><span>Sharp focus, not blurry</span></li>
        </ul>
      </div>
      {/* An unfamiliar photo gets no guesses at all: any guess could mislead. */}
      {!unfamiliar && (
        <details className="breakdown">
          <summary>Possible, but not reliable</summary>
          <Breakdown result={result} muted />
        </details>
      )}
    </>
  )
}

function ConfidenceMeter({ value, threshold }: { value: number; threshold: number }) {
  const level = value >= 0.95 ? 'High' : value >= 0.85 ? 'Good' : 'Moderate'
  return (
    <div className="meter">
      <div className="meter__label">
        <span>Confidence</span>
        <strong>
          {pct(value)} <span className="meter__level">{level}</span>
        </strong>
      </div>
      <div
        className="meter__track"
        role="progressbar"
        aria-label="Confidence"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
      >
        <div className="meter__fill" style={{ width: pct(value) }} />
        <div className="meter__tick" style={{ left: pct(threshold) }} />
      </div>
      <p className="meter__hint">The app only gives an answer above {pct(threshold)} confidence.</p>
    </div>
  )
}

function Breakdown({ result, muted = false }: { result: Prediction; muted?: boolean }) {
  return (
    <ol className={`bars${muted ? ' bars--muted' : ''}`} aria-label="All answers by confidence">
      {result.top3.map((g) => (
        <li key={g.label} className={`bars__row bars__row--${g.label}`}>
          <span className="bars__name">{displayName(g.label)}</span>
          <span className="bars__track">
            <span style={{ width: pct(g.confidence) }} />
          </span>
          <span className="bars__pct">{pct(g.confidence)}</span>
        </li>
      ))}
    </ol>
  )
}

import { AlertIcon, ArrowRightIcon, LockIcon, OutcomeIcon, ShieldIcon } from '../components/Icons'
import { OUTCOMES, type Outcome } from '../labels'

const ORDER: Outcome[] = ['rust', 'other', 'healthy', 'not_sure']

const RESULTS = [
  { set: 'Field photos from farms kept out of training (Ecuador)', n: '405', acc: '99%' },
  { set: 'Held-out test photos, all five datasets', n: '603', acc: '96%' },
  { set: 'Other leaf diseases wrongly called rust', n: '603', acc: '1.3%' },
]

const DATASETS = [
  { name: 'RoCoLe', where: 'Ecuador', what: 'Robusta leaves photographed on the plant with a phone, in the field' },
  { name: 'BRACOL', where: 'Brazil', what: 'Arabica leaves photographed on a white background' },
  { name: 'JMuBEN and JMuBEN2', where: 'Kenya', what: 'Close-up crops of Arabica leaves' },
  { name: 'Uganda coffee leaf dataset', where: 'Uganda', what: 'Close-ups of part of a leaf; coffee variety not stated' },
]

export function AboutPage() {
  return (
    <main>
      <section className="hero hero--compact">
        <div className="hero__inner">
          <p className="eyebrow eyebrow--light">About Coffee Leaf AI</p>
          <h1>An early-warning screen for coffee leaf rust</h1>
          <p className="hero__sub">
            Coffee leaf rust is one of the most damaging diseases of coffee. Spotting it early helps farmers act before
            it spreads. Coffee Leaf AI is a prototype that lets anyone with a phone screen a leaf in seconds.
          </p>
        </div>
      </section>

      <div className="doc">
        <section className="doc__section">
          <h2>What it does</h2>
          <p>
            It checks a photo of a <strong>single coffee leaf</strong> and gives one of four answers. Answers are
            worded as screening results, not diagnoses.
          </p>
          <ul className="outcome-list">
            {ORDER.map((o) => (
              <li key={o} className={`outcome-list__item outcome outcome--${o}`}>
                <span className="outcome__icon"><OutcomeIcon outcome={o} /></span>
                <div>
                  <strong>{OUTCOMES[o].title}</strong>
                  <p>{OUTCOMES[o].short}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="doc__section">
          <h2>How reliable is it?</h2>
          <p>
            All results below are on photos the model never saw while learning. It only answers when it is at least
            77% confident and the photo looks like the photos it learned from; otherwise it says "not sure".
          </p>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th scope="col">Test</th><th scope="col">Photos</th><th scope="col">Result</th></tr>
              </thead>
              <tbody>
                {RESULTS.map((r) => (
                  <tr key={r.set}><td>{r.set}</td><td>{r.n}</td><td><strong>{r.acc}</strong></td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="callout callout--warn">
            <AlertIcon />
            <span>
              <strong>Accuracy depends on the photo style.</strong> When the model was tested on a whole dataset it had
              not trained on, accuracy dropped considerably. Real farmer photos may differ from the training photos,
              so treat every answer as a hint and confirm it.
            </span>
          </p>
        </section>

        <section className="doc__section">
          <h2>What it cannot do</h2>
          <ul className="bullets">
            <li>
              <strong>It does not detect pests or most other problems.</strong> Mite damage, nutrient deficiencies and
              many diseases are unknown to it, and some of them can be mistaken for rust.
            </li>
            <li>It looks at one leaf at a time. A leaf with several problems may get a misleading answer.</li>
            <li>For problems other than rust it does not say which disease it is.</li>
            <li>It has not yet been tested on photos taken by farmers using the app.</li>
            <li>It is not a diagnosis. Confirm with an agronomist or extension officer before treating.</li>
          </ul>
        </section>

        <section className="doc__section">
          <h2>Data and credits</h2>
          <p>
            The model was trained on five public, expert-labelled datasets covering Arabica and Robusta coffee. Photos
            were de-duplicated and whole farms were held back for testing, so the scores above reflect new leaves
            rather than near-copies of training photos.
          </p>
          <ul className="datasets">
            {DATASETS.map((d) => (
              <li key={d.name}>
                <strong>{d.name}</strong>
                <span className="datasets__where">{d.where}</span>
                <span>{d.what}</span>
              </li>
            ))}
          </ul>
          <p className="muted small">
            Example photos in the app come from the held-out test sets of RoCoLe and BRACOL. We thank the authors of
            all five datasets for making their work openly available.
          </p>
        </section>

        <section className="doc__section doc__grid">
          <div className="infobox">
            <LockIcon />
            <h3>Privacy</h3>
            <p>
              Photos are sent to our server only to be analysed, and are discarded straight after. The app does not
              store photos, ask for an account, or track who uses it.
            </p>
          </div>
          <div className="infobox">
            <ShieldIcon />
            <h3>Responsible use</h3>
            <p>
              The app is designed to say "not sure" rather than guess, and every answer carries a confidence score so
              people can judge how much to rely on it.
            </p>
          </div>
        </section>

        <section className="doc__section maker">
          <img className="maker__logo" src="/brand/aplyd-logo.png" alt="APLYD" width="295" height="120" />
          <div>
            <h2>Who made it</h2>
            <p>
              Coffee Leaf AI is a prototype developed by <strong>APLYD</strong>, by Athena Infonomics.
            </p>
          </div>
        </section>

        <a className="btn btn--primary btn--lg" href="#/">
          Check a leaf now <ArrowRightIcon />
        </a>
      </div>
    </main>
  )
}

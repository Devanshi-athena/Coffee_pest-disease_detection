import { Analyzer } from '../components/Analyzer'
import {
  ArrowRightIcon,
  CameraIcon,
  CheckIcon,
  FocusIcon,
  GlobeIcon,
  LeafIcon,
  OutcomeIcon,
  ScanIcon,
  ShieldIcon,
  SunIcon,
  XIcon,
} from '../components/Icons'
import { OUTCOMES, type Outcome } from '../labels'
import { useServerStatus } from '../useServerStatus'

const ORDER: Outcome[] = ['rust', 'other', 'healthy', 'not_sure']

export function HomePage() {
  const server = useServerStatus()

  return (
    <main>
      <section className="hero">
        <div className="hero__inner">
          <p className="pill">
            <LeafIcon /> AI screening for coffee leaf rust
          </p>
          <h1>
            Spot coffee leaf rust <span className="hero__accent">from a single photo.</span>
          </h1>
          <p className="hero__sub">
            Coffee Leaf AI looks at a photo of a coffee leaf and tells you, in about a second, whether it shows signs
            of leaf rust, another leaf problem, or looks healthy. When it is unsure, it says so.
          </p>
        </div>
      </section>

      <section className="workbench" id="check" aria-label="Check a leaf">
        <Analyzer server={server} />
      </section>

      <section className="section" id="how-it-works" aria-labelledby="how-title">
        <div className="section__head">
          <p className="eyebrow">How it works</p>
          <h2 id="how-title">Three steps, no expertise needed</h2>
        </div>
        <ol className="steps">
          <li className="step">
            <span className="step__icon"><CameraIcon /></span>
            <span className="step__num">1</span>
            <h3>Photograph a leaf</h3>
            <p>Use any phone camera. One leaf, filling most of the frame, in daylight.</p>
          </li>
          <li className="step">
            <span className="step__icon"><ScanIcon /></span>
            <span className="step__num">2</span>
            <h3>The AI checks it</h3>
            <p>
              A model trained on thousands of labelled leaf photos from four countries compares your photo with what
              it has learned.
            </p>
          </li>
          <li className="step">
            <span className="step__icon"><CheckIcon /></span>
            <span className="step__num">3</span>
            <h3>Get a clear answer</h3>
            <p>One of four plain-language answers, a confidence score and practical next steps.</p>
          </li>
        </ol>
      </section>

      <section className="section section--tint" aria-labelledby="answers-title">
        <div className="wrap">
        <div className="section__head">
          <p className="eyebrow">What it can tell you</p>
          <h2 id="answers-title">Four possible answers</h2>
          <p className="section__lead">
            The app is designed to be honest about its limits: it prefers saying "not sure" over guessing.
          </p>
        </div>
        <ul className="outcomes">
          {ORDER.map((o) => (
            <li key={o} className={`outcome outcome--${o}`}>
              <span className="outcome__icon"><OutcomeIcon outcome={o} /></span>
              <h3>{OUTCOMES[o].title}</h3>
              <p>{OUTCOMES[o].short}</p>
            </li>
          ))}
        </ul>
        </div>
      </section>

      <section className="section" aria-labelledby="tips-title">
        <div className="section__head">
          <p className="eyebrow">Better photos, better answers</p>
          <h2 id="tips-title">How to take a good photo</h2>
        </div>
        <div className="tips">
          <div className="tips__col tips__col--do">
            <h3><CheckIcon /> Do</h3>
            <ul>
              <li><LeafIcon /> One leaf, filling most of the frame</li>
              <li><SunIcon /> Natural daylight, in open shade</li>
              <li><FocusIcon /> Hold steady and tap to focus on the leaf</li>
              <li><ScanIcon /> Show the spots clearly, top or underside</li>
            </ul>
          </div>
          <div className="tips__col tips__col--dont">
            <h3><XIcon /> Avoid</h3>
            <ul>
              <li><XIcon /> Several leaves or the whole plant in one photo</li>
              <li><XIcon /> Flash glare, harsh sun or deep shadow</li>
              <li><XIcon /> Blurry or very dark photos</li>
              <li><XIcon /> Leaves of other crops: it only knows coffee</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="trust" aria-labelledby="trust-title">
        <div className="trust__inner">
          <div className="section__head section__head--light">
            <p className="eyebrow">Built to be trusted</p>
            <h2 id="trust-title">Tested on photos it had never seen</h2>
          </div>
          <dl className="stats">
            <div className="stat">
              <dt>99%</dt>
              <dd>accuracy on field photos from coffee farms that were kept out of training</dd>
            </div>
            <div className="stat">
              <dt>96%</dt>
              <dd>accuracy on held-out test photos from all five datasets</dd>
            </div>
            <div className="stat">
              <dt>4</dt>
              <dd>countries in the training data: Kenya, Uganda, Brazil and Ecuador</dd>
            </div>
            <div className="stat">
              <dt>&lt;1 s</dt>
              <dd>typical time to an answer once the server is running</dd>
            </div>
          </dl>
          <div className="trust__notes">
            <p><ShieldIcon /> Photos are analysed and then discarded. They are not stored.</p>
            <p><GlobeIcon /> Accuracy can be lower on photo styles and regions it was not trained on.</p>
          </div>
          <a className="btn btn--gold" href="#/about">
            Read about the model and its limits <ArrowRightIcon />
          </a>
        </div>
      </section>
    </main>
  )
}

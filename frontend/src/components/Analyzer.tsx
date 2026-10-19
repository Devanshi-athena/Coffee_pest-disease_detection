import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { ApiError, predict, type Prediction } from '../api'
import { shrinkImage } from '../image'
import { outcomeOf } from '../labels'
import { loadSample, SAMPLES, sampleThumb, type Sample } from '../samples'
import type { ServerStatus } from '../useServerStatus'
import { CameraIcon, ClipboardIcon, RefreshIcon, ScanIcon, UploadIcon } from './Icons'
import { ResultCard } from './ResultCard'

type State =
  | { status: 'idle' }
  | { status: 'loading'; slow: boolean }
  | { status: 'done'; result: Prediction }
  | { status: 'error'; message: string }

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']
const SLOW_MS = 6000

const STATUS_TEXT: Record<ServerStatus, string> = {
  checking: 'Connecting…',
  waking: 'Starting up…',
  ready: 'Model ready',
  no_model: 'Model unavailable',
  offline: 'Server offline',
}

// Phones get a "Take photo" button that opens the camera directly.
const touchDevice = () => window.matchMedia?.('(pointer: coarse)').matches ?? false

export function Analyzer({ server }: { server: ServerStatus }) {
  const fileInput = useRef<HTMLInputElement>(null)
  const cameraInput = useRef<HTMLInputElement>(null)
  const inFlight = useRef<AbortController | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [source, setSource] = useState<string | null>(null)
  const [state, setState] = useState<State>({ status: 'idle' })
  const [dragging, setDragging] = useState(false)
  const [canCapture] = useState(touchDevice)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  async function analyze(file: File, label: string | null = null) {
    inFlight.current?.abort()
    if (!ACCEPTED.includes(file.type)) {
      setPreview(null)
      setSource(null)
      setState({ status: 'error', message: 'Please choose a JPEG, PNG or WebP photo.' })
      return
    }
    const controller = new AbortController()
    inFlight.current = controller
    setPreview(URL.createObjectURL(file))
    setSource(label)
    setState({ status: 'loading', slow: false })
    const slowTimer = window.setTimeout(() => {
      if (!controller.signal.aborted) setState((s) => (s.status === 'loading' ? { ...s, slow: true } : s))
    }, SLOW_MS)
    try {
      const image = await shrinkImage(file)
      const result = await predict(image, file.name || 'leaf.jpg', controller.signal)
      setState({ status: 'done', result })
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      const message = e instanceof ApiError ? e.message : 'Something went wrong. Please try again.'
      setState({ status: 'error', message })
    } finally {
      window.clearTimeout(slowTimer)
    }
  }

  // Paste a photo from the clipboard (handy when presenting from a laptop).
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith('image/'))
      const file = item?.getAsFile()
      if (file) {
        e.preventDefault()
        void analyze(file, 'Pasted image')
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  })

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = '' // allow picking the same file again
    if (file) void analyze(file, file.name)
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) void analyze(file, file.name)
  }

  async function onSample(s: Sample) {
    try {
      await analyze(await loadSample(s), `Example: ${s.title} (${s.caption}, ${s.credit} dataset)`)
    } catch (e) {
      setState({ status: 'error', message: (e as Error).message })
    }
  }

  function reset() {
    inFlight.current?.abort()
    setPreview(null)
    setSource(null)
    setState({ status: 'idle' })
  }

  // On narrow screens the answer sits below the photo: bring it into view when it arrives.
  const panelRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (state.status !== 'done' && state.status !== 'error') return
    if (window.matchMedia?.('(max-width: 860px)').matches) {
      panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [state.status])

  const busy = state.status === 'loading'
  const outcome = state.status === 'done' ? outcomeOf(state.result) : null
  const dragProps = {
    onDragOver: (e: DragEvent) => { e.preventDefault(); setDragging(true) },
    onDragLeave: () => setDragging(false),
    onDrop,
  }

  return (
    <div className={`analyzer${dragging ? ' analyzer--drag' : ''}${preview ? '' : ' analyzer--idle'}`} {...dragProps}>
      <input ref={fileInput} type="file" accept={ACCEPTED.join(',')} hidden onChange={onPick} />
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={onPick} />

      {/* Left: the photo */}
      <div className="stage">
        {!preview ? (
          <button type="button" className="dropzone" onClick={() => fileInput.current?.click()}>
            <span className="dropzone__icon"><UploadIcon /></span>
            <span className="dropzone__title">Drop a leaf photo here</span>
            <span className="dropzone__hint">or click to browse · JPEG, PNG or WebP</span>
            <span className="dropzone__paste"><ClipboardIcon /> You can also paste an image</span>
          </button>
        ) : (
          <figure className={`photo${outcome ? ` photo--${outcome}` : ''}`}>
            <img src={preview} alt="The leaf photo being checked" />
            {busy && (
              <div className="photo__scan" aria-hidden="true">
                <span className="photo__line" />
              </div>
            )}
            {source && <figcaption>{source}</figcaption>}
          </figure>
        )}
      </div>

      {/* Right: actions, progress or the answer */}
      <div className="panel" ref={panelRef}>
        {state.status === 'idle' && (
          <div className="panel__idle">
            <div className="panel__top">
              <h2>Check a leaf</h2>
              <span className={`status status--${server}`} role="status">
                <span className="status__dot" aria-hidden="true" />
                {STATUS_TEXT[server]}
              </span>
            </div>
            <p className="muted">
              Photograph <strong>one coffee leaf</strong> filling most of the frame, in daylight. Results take about
              a second.
            </p>
            <div className="actions">
              {canCapture && (
                <button type="button" className="btn btn--primary" onClick={() => cameraInput.current?.click()}>
                  <CameraIcon /> Take a photo
                </button>
              )}
              <button
                type="button"
                className={`btn ${canCapture ? 'btn--secondary' : 'btn--primary'}`}
                onClick={() => fileInput.current?.click()}
              >
                <UploadIcon /> Upload a photo
              </button>
            </div>
            <div className="examples">
              <p className="examples__label">No leaf at hand? Try an example</p>
              <ul className="examples__grid">
                {SAMPLES.map((s) => (
                  <li key={s.id}>
                    <button type="button" className="example" onClick={() => void onSample(s)}>
                      <img src={sampleThumb(s)} alt="" loading="lazy" width="160" height="160" />
                      <span className="example__title">{s.title}</span>
                      <span className="example__caption">{s.caption}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {state.status === 'loading' && (
          <div className="panel__loading" role="status" aria-live="polite">
            <span className="loader" aria-hidden="true"><ScanIcon /></span>
            <h2>Analysing the leaf…</h2>
            <p className="muted">
              {state.slow
                ? 'The server is waking up after a quiet period. The first check can take up to a minute; after that, results are almost instant.'
                : 'Looking for signs of rust and other leaf problems.'}
            </p>
            <button type="button" className="btn btn--ghost" onClick={reset}>Cancel</button>
          </div>
        )}

        {state.status === 'error' && (
          <div className="panel__error" role="alert">
            <h2>We couldn't check this photo</h2>
            <p>{state.message}</p>
            <button type="button" className="btn btn--primary" onClick={reset}>
              <RefreshIcon /> Try again
            </button>
          </div>
        )}

        {state.status === 'done' && (
          <>
            <ResultCard result={state.result} />
            <div className="actions">
              <button type="button" className="btn btn--primary" onClick={() => { reset(); fileInput.current?.click() }}>
                <UploadIcon /> Check another leaf
              </button>
              <button type="button" className="btn btn--secondary" onClick={reset}>
                Start over
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

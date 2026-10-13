import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { ApiError, predict, type Prediction } from '../api'
import { ResultCard } from '../components/ResultCard'
import { shrinkImage } from '../image'

type State =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done'; result: Prediction }
  | { status: 'error'; message: string }

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

export function PredictPage() {
  const fileInput = useRef<HTMLInputElement>(null)
  const inFlight = useRef<AbortController | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [state, setState] = useState<State>({ status: 'idle' })
  const [dragging, setDragging] = useState(false)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  async function analyze(file: File) {
    inFlight.current?.abort()
    if (!ACCEPTED.includes(file.type)) {
      setPreview(null)
      setState({ status: 'error', message: 'Please choose a JPEG, PNG or WebP image.' })
      return
    }
    const controller = new AbortController()
    inFlight.current = controller
    setPreview(URL.createObjectURL(file))
    setState({ status: 'loading' })
    try {
      const image = await shrinkImage(file)
      const result = await predict(image, file.name || 'leaf.jpg', controller.signal)
      setState({ status: 'done', result })
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      const message = e instanceof ApiError ? e.message : 'Something went wrong. Please try again.'
      setState({ status: 'error', message })
    }
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = '' // allow picking the same file again
    if (file) void analyze(file)
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) void analyze(file)
  }

  // Clear the previous result and open the file picker straight away.
  function checkAnother() {
    inFlight.current?.abort()
    setPreview(null)
    setState({ status: 'idle' })
    fileInput.current?.click()
  }

  const busy = state.status === 'loading'
  const finished = state.status === 'done' || state.status === 'error'

  return (
    <main className="page">
      <p className="lead">
        Upload a clear photo of <strong>one coffee leaf</strong> filling most of the frame, in daylight, to check it
        for signs of coffee leaf rust.
      </p>

      <input ref={fileInput} type="file" accept={ACCEPTED.join(',')} hidden onChange={onPick} />

      {!preview && (
        <button
          type="button"
          className={`dropzone${dragging ? ' dropzone--active' : ''}`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <svg className="dropzone__icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
          <span className="dropzone__title">Upload image</span>
          <span className="dropzone__hint">Tap to choose, or drag a photo here. JPEG, PNG or WebP.</span>
        </button>
      )}

      {preview && (
        <figure className="preview">
          <img src={preview} alt="Selected leaf" />
          {busy && (
            <div className="preview__overlay" role="status">
              <span className="spinner" aria-hidden="true" />
              Analyzing…
            </div>
          )}
        </figure>
      )}

      {state.status === 'error' && (
        <div className="card card--error" role="alert">
          <p>{state.message}</p>
        </div>
      )}

      {state.status === 'done' && <ResultCard result={state.result} />}

      {finished && (
        <button type="button" className="btn btn--primary" onClick={checkAnother}>
          Check another image
        </button>
      )}
    </main>
  )
}

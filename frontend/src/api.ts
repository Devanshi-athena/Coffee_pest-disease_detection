export type Guess = { label: string; confidence: number }

export type Prediction = {
  label: string
  confidence: number
  top3: Guess[]
  uncertain: boolean
  reason?: 'low_confidence' | 'unfamiliar_image' | null
  outcome?: string
  threshold: number
  distance?: number | null
}

// Unset: local backend. Set to an empty string: same origin (backend also serves the frontend, e.g. Hugging Face).
export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8100').replace(/\/+$/, '')

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

const FRIENDLY: Record<number, string> = {
  503: 'The model is not loaded on the server yet. Try again later.',
  413: 'That photo is too large. Try a smaller one.',
  415: 'That file type is not supported. Use a JPEG, PNG or WebP photo.',
}

export async function predict(image: Blob, filename: string, signal?: AbortSignal): Promise<Prediction> {
  const form = new FormData()
  form.append('file', image, filename)
  let res: Response
  try {
    res = await fetch(`${API_URL}/predict`, { method: 'POST', body: form, signal })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(`Cannot reach the server at ${API_URL}. Check your connection.`, 0)
  }
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json()).detail ?? ''
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(FRIENDLY[res.status] ?? (detail || `Request failed (${res.status}).`), res.status)
  }
  return res.json()
}

export type Health = { status: string; model_loaded: boolean }

/** Pings the server. Also wakes a sleeping free-tier host before the first photo is sent. */
export async function health(signal?: AbortSignal): Promise<Health> {
  const res = await fetch(`${API_URL}/health`, { signal, cache: 'no-store' })
  if (!res.ok) throw new ApiError(`Health check failed (${res.status}).`, res.status)
  return res.json()
}

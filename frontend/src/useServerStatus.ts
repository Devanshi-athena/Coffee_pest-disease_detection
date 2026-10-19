import { useEffect, useState } from 'react'
import { health } from './api'

export type ServerStatus = 'checking' | 'waking' | 'ready' | 'no_model' | 'offline'

// Free hosting puts the server to sleep when idle; the first request can take up to a minute.
// Ping it as soon as the page opens so it is usually awake by the time someone picks a photo.
const SLOW_AFTER_MS = 2500
const RETRY_MS = 4000
const GIVE_UP_AFTER_MS = 90_000

export function useServerStatus(): ServerStatus {
  const [status, setStatus] = useState<ServerStatus>('checking')

  useEffect(() => {
    let cancelled = false
    const started = Date.now()
    const controller = new AbortController()
    const slow = window.setTimeout(() => !cancelled && setStatus((s) => (s === 'checking' ? 'waking' : s)), SLOW_AFTER_MS)
    let retry: number | undefined

    async function ping() {
      try {
        const h = await health(controller.signal)
        if (cancelled) return
        setStatus(h.model_loaded ? 'ready' : 'no_model')
      } catch (e) {
        if (cancelled || (e as Error).name === 'AbortError') return
        if (Date.now() - started > GIVE_UP_AFTER_MS) {
          setStatus('offline')
          return
        }
        setStatus('waking')
        retry = window.setTimeout(ping, RETRY_MS)
      }
    }
    void ping()

    return () => {
      cancelled = true
      controller.abort()
      window.clearTimeout(slow)
      window.clearTimeout(retry)
    }
  }, [])

  return status
}

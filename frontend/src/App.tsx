import { useEffect, useState } from 'react'
import { AboutPage } from './pages/AboutPage'
import { PredictPage } from './pages/PredictPage'

// Tiny hash router: "#/about" or anything else -> predict page. Works on any static host.
function useHashRoute() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const onChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return hash === '#/about' ? 'about' : 'predict'
}

export default function App() {
  const route = useHashRoute()
  return (
    <>
      <header className="topbar">
        <a href="#/" className="brand">
          Coffee Leaf AI
        </a>
        <nav>
          <a href="#/" aria-current={route === 'predict' ? 'page' : undefined}>
            Check leaf
          </a>
          <a href="#/about" aria-current={route === 'about' ? 'page' : undefined}>
            About
          </a>
        </nav>
      </header>
      {route === 'about' ? <AboutPage /> : <PredictPage />}
    </>
  )
}

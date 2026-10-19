import { useEffect, useState } from 'react'
import { AboutPage } from './pages/AboutPage'
import { HomePage } from './pages/HomePage'

type Route = 'home' | 'how' | 'about'

// Tiny hash router: "#/about", "#/how-it-works" (a section of the home page) or anything else -> home.
// Works on any static host.
function parse(hash: string): Route {
  if (hash === '#/about') return 'about'
  if (hash === '#/how-it-works') return 'how'
  return 'home'
}

function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parse(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

const TITLES: Record<Route, string> = {
  home: 'Coffee Leaf AI | APLYD',
  how: 'How it works | Coffee Leaf AI',
  about: 'About | Coffee Leaf AI',
}

export default function App() {
  const route = useHashRoute()

  useEffect(() => {
    document.title = TITLES[route]
    if (route === 'how') {
      document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    } else {
      window.scrollTo({ top: 0 })
    }
  }, [route])

  return (
    <>
      <a className="skip" href="#main-content">Skip to content</a>
      <header className="topbar">
        <div className="topbar__inner">
          <a href="#/" className="brand" aria-label="APLYD Coffee Leaf AI, home">
            <picture>
              <source srcSet="/brand/aplyd-logo-light.png" media="(prefers-color-scheme: dark)" />
              <img className="brand__logo" src="/brand/aplyd-logo.png" alt="APLYD" width="295" height="120" />
            </picture>
            <span className="brand__app">Coffee Leaf AI</span>
          </a>
          <nav className="nav" aria-label="Main">
            <a href="#/" aria-current={route === 'home' ? 'page' : undefined}>Check a leaf</a>
            <a href="#/how-it-works" aria-current={route === 'how' ? 'page' : undefined}>How it works</a>
            <a href="#/about" aria-current={route === 'about' ? 'page' : undefined}>About</a>
          </nav>
          <span className="tag" title="This is a prototype, not a finished product">Prototype</span>
        </div>
      </header>

      <div id="main-content">{route === 'about' ? <AboutPage /> : <HomePage />}</div>

      <footer className="footer">
        <div className="footer__inner">
          <div className="footer__brand">
            <img src="/brand/aplyd-logo-light.png" alt="APLYD" width="295" height="120" />
            <p>Coffee Leaf AI is a prototype by APLYD, Athena Infonomics.</p>
          </div>
          <nav className="footer__nav" aria-label="Footer">
            <a href="#/">Check a leaf</a>
            <a href="#/how-it-works">How it works</a>
            <a href="#/about">About the model</a>
          </nav>
          <p className="footer__legal">
            A screening aid, not a diagnosis. Always confirm with an agronomist or extension officer before treating.
          </p>
        </div>
      </footer>
    </>
  )
}

// Small inline icon set (stroke icons, 24px grid). Decorative: always aria-hidden.
import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement>

function Svg({ children, ...rest }: P) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="24"
      height="24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  )
}

export const UploadIcon = (p: P) => (
  <Svg {...p}><path d="M12 15V4m0 0-4 4m4-4 4 4M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></Svg>
)
export const CameraIcon = (p: P) => (
  <Svg {...p}>
    <path d="M4 8a2 2 0 0 1 2-2h1.5l1.4-2h6.2l1.4 2H18a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z" />
    <circle cx="12" cy="12.5" r="3.5" />
  </Svg>
)
export const LeafIcon = (p: P) => (
  <Svg {...p}>
    <path d="M20 4C10 4 4 9 4 16c0 1.4.3 2.6.9 3.6M20 4c0 9-4.5 15-11.5 15-1.5 0-2.8-.3-3.6-.8M20 4 9 15" />
  </Svg>
)
export const ScanIcon = (p: P) => (
  <Svg {...p}>
    <path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M4 12h16" />
  </Svg>
)
export const ClipboardIcon = (p: P) => (
  <Svg {...p}>
    <path d="M9 4h6v3H9zM9 5.5H6.5A1.5 1.5 0 0 0 5 7v12.5A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V7a1.5 1.5 0 0 0-1.5-1.5H15M9 12h6M9 16h4" />
  </Svg>
)
export const CheckIcon = (p: P) => <Svg {...p}><path d="m5 12.5 4.5 4.5L19 7.5" /></Svg>
export const XIcon = (p: P) => <Svg {...p}><path d="M6 6l12 12M18 6 6 18" /></Svg>
export const AlertIcon = (p: P) => (
  <Svg {...p}><path d="M12 4 2.8 19.5h18.4Z" /><path d="M12 10v4.2M12 17.2v.1" /></Svg>
)
export const SpotIcon = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="8.5" /><circle cx="9.5" cy="10" r="1.6" /><circle cx="14.5" cy="14" r="2" /></Svg>
)
export const HelpIcon = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="8.5" /><path d="M9.6 9.5a2.5 2.5 0 0 1 4.8 1c0 1.7-2.4 2-2.4 3.5M12 16.8v.1" /></Svg>
)
export const ShieldIcon = (p: P) => (
  <Svg {...p}><path d="M12 3.5 5 6v5.5c0 4.3 3 7.6 7 9 4-1.4 7-4.7 7-9V6Z" /><path d="m9 12 2.2 2.2L15.5 10" /></Svg>
)
export const GlobeIcon = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="8.5" /><path d="M3.5 12h17M12 3.5c2.3 2.4 3.5 5.2 3.5 8.5s-1.2 6.1-3.5 8.5c-2.3-2.4-3.5-5.2-3.5-8.5s1.2-6.1 3.5-8.5Z" /></Svg>
)
export const ArrowRightIcon = (p: P) => <Svg {...p}><path d="M5 12h14m-5-5 5 5-5 5" /></Svg>
export const RefreshIcon = (p: P) => (
  <Svg {...p}><path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4" /></Svg>
)
export const SunIcon = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="3.8" /><path d="M12 2.8v2M12 19.2v2M4.4 4.4l1.4 1.4M18.2 18.2l1.4 1.4M2.8 12h2M19.2 12h2M4.4 19.6l1.4-1.4M18.2 5.8l1.4-1.4" /></Svg>
)
export const FocusIcon = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="3" /><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16" /></Svg>
)
export const LockIcon = (p: P) => (
  <Svg {...p}><rect x="5" y="10.5" width="14" height="9.5" rx="2" /><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5" /></Svg>
)

/** Icon for each answer, so the meaning never relies on colour alone. */
export function OutcomeIcon({ outcome, ...p }: P & { outcome: string }) {
  if (outcome === 'rust') return <AlertIcon {...p} />
  if (outcome === 'other') return <SpotIcon {...p} />
  if (outcome === 'healthy') return <CheckIcon {...p} />
  return <HelpIcon {...p} />
}

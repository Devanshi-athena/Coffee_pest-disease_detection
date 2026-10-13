// Display names for the model's class ids (backend /classes, Phase 2 model: healthy / rust / other).
export const LABELS: Record<string, string> = {
  healthy: 'Looks healthy',
  rust: 'Rust-like symptoms',
  other: 'Another leaf problem (not rust)',
}

export const displayName = (id: string) => LABELS[id] ?? id

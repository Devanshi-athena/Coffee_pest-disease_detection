// Example photos for trying the app without a leaf at hand (e.g. during a demo).
// All are held-out test photos from public datasets: the model never trained on them (see trial_images/README.md).

export type Sample = {
  id: string
  title: string
  caption: string
  credit: string
}

export const SAMPLES: Sample[] = [
  { id: 'healthy-field', title: 'Healthy leaf', caption: 'Field photo, Ecuador', credit: 'RoCoLe' },
  { id: 'rust-field', title: 'Leaf rust', caption: 'Field photo, Ecuador', credit: 'RoCoLe' },
  { id: 'rust-sheet', title: 'Leaf rust', caption: 'Lab photo, Brazil', credit: 'BRACOL' },
  { id: 'phoma-sheet', title: 'Phoma leaf spot', caption: 'Lab photo, Brazil', credit: 'BRACOL' },
  { id: 'mite-field', title: 'Spider mite damage', caption: 'Not in training data', credit: 'RoCoLe' },
]

export const sampleThumb = (s: Sample) => `/samples/thumbs/${s.id}.jpg`
export const samplePhoto = (s: Sample) => `/samples/${s.id}.jpg`

export async function loadSample(s: Sample): Promise<File> {
  const res = await fetch(samplePhoto(s))
  if (!res.ok) throw new Error(`Could not load the example photo (${res.status}).`)
  const blob = await res.blob()
  return new File([blob], `${s.id}.jpg`, { type: blob.type || 'image/jpeg' })
}

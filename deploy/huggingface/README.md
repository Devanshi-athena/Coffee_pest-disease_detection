---
title: Coffee Leaf AI
emoji: 🍃
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Screen a coffee leaf photo for rust. APLYD prototype.
---

# Coffee Leaf AI (APLYD prototype)

Upload a photo of one coffee leaf. The app answers **Rust-like symptoms**, **Looks healthy**,
**Another leaf problem (not rust)** or **Not sure**.

A screening aid, not a diagnosis: confirm with an agronomist or extension officer before treating.

- Model: ConvNeXt-Tiny fine-tuned on five public coffee leaf datasets (Kenya, Uganda, Brazil, Ecuador),
  served with onnxruntime on CPU. 99% accuracy on field photos from farms it never saw, about 96% on
  held-out test photos; much lower on photo styles unlike its training data.
- Unknown problems (for example mite damage) are often answered as rust.
- Photos are processed in memory and not stored by the app.

Developed by APLYD, by Athena Infonomics.

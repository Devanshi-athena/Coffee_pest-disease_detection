export function AboutPage() {
  return (
    <main className="page prose">
      <h2>About</h2>
      <p>
        Coffee Leaf AI checks a photo of a <strong>single coffee leaf</strong> for signs of{' '}
        <strong>coffee leaf rust</strong>. It gives one of four answers:
      </p>
      <ul>
        <li><strong>Rust-like symptoms</strong>: signs that look like coffee leaf rust.</li>
        <li><strong>Looks healthy</strong>: no signs of rust or the other leaf diseases it knows.</li>
        <li>
          <strong>Another leaf problem (not rust)</strong>: it recognises leaf spots (phoma, cercospora) and leaf
          miner damage as "not rust", but does not say which one.
        </li>
        <li>
          <strong>Not sure</strong>: the photo is unclear, or it looks unlike the photos the model learned from.
          It prefers saying "not sure" over guessing.
        </li>
      </ul>

      <h3>How reliable is it?</h3>
      <ul>
        <li>
          On field photos of whole leaves from farms it had never seen (same kind of photos as part of its
          training data), it was right about 99% of the time when it gave an answer.
        </li>
        <li>
          On photo styles it was never trained on, tests showed much lower accuracy. Your photos may differ from
          the training photos, so treat every answer as a hint, not a result.
        </li>
      </ul>

      <h3>Limitations</h3>
      <ul>
        <li>
          <strong>It does not detect pests or most other problems.</strong> Mite damage, nutrient deficiencies and
          many diseases are unknown to it; some of them can be mistaken for rust.
        </li>
        <li>One leaf per photo. Leaves with several problems may get a misleading answer.</li>
        <li>Trained on public datasets from Kenya, Uganda, Brazil and Ecuador (Arabica and Robusta).</li>
        <li>It is not a diagnosis. Confirm with an agronomist or extension officer before treating.</li>
      </ul>

      <h3>Who made it</h3>
      <p>
        Coffee Leaf AI is a prototype developed by <strong>APLYD</strong>, by Athena Infonomics.
      </p>

      <h3>Privacy</h3>
      <p>Photos are sent to the server for analysis and are not stored by this app.</p>
    </main>
  )
}

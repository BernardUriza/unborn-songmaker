# samples/

Recorded audio for the sampler voice. A Track with `voice: "sample:<name>"`
triggers `samples/<name>.wav`, pitched by the track's `note` (note 60 = native
pitch, higher notes play faster/higher). Because it is an ordinary Track, the
sample inherits polymeter, swing and the modulation network.

## Recording spec (Korda-style spoken layer)

- **Short and rhythmic.** Single words or 1-2s phrases loop best ("unborn",
  "generate", a breath, a count). Korda's vocals are spoken fragments, repeated.
- **Mono, close mic, minimal background.** A phone voice memo is fine.
- **Any format in** (m4a/wav/mp3) -- it gets converted to mono 44.1k wav.
- **One file per word**, named by the word: `samples/unborn.wav`.

Then reference it in a spec: `{ "voice": "sample:unborn", "note": 60, ... }`.

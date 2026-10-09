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

## Foley takes (home SFX from videopipe)

`python cli.py import-sfx ~/Downloads/sfx-<session>.zip` unpacks a bundle from
videopipe's recorder (`/sfx/record/<session>`) into one folder per sound:

    samples/stamp_hit/take-01.wav, take-04.wav, ... + takes.json (provenance)

`"voice": "sample:stamp_hit"` then plays a different take on each trigger
(round-robin), pitched by `note` like any sample. Only the takes marked
"conservar" in the recorder are imported; the leading ~0.5 s of room tone is cut
so the transient lands on the beat. A `samples/<name>.wav` file, if present,
wins over the folder of the same name.

# unborn-songmaker

> *Thou shalt not procreate. Thou shalt, however, generate.*

A rules-based generative sound engine. You describe a sound or a sculpture in
plain words, Claude writes the spec, the engine renders an **mp3 ready to drop
into an app**. No neural black box: every waveform is summed sine partials over a
readable envelope, and every piece of music emerges from a network of rules you
can inspect and edit.

Built in homage to **Chris Korda** -- reverend of the Church of Euthanasia,
pioneer of complex polymeter -- whose [Polymeter MIDI Sequencer](https://github.com/victimofleisure/Polymeter)
(GPL-3.0) is the model this engine reimplements. Korda builds invisible kinetic
sculptures and the sculptures generate the music; her sequencer emits MIDI to
external synths. `unborn-songmaker` ports that *technique* (a clean-room
reimplementation of the algorithm, which is not copyrightable) and adds the two
things her tool deliberately leaves out: a built-in synthesizer and an audio
export. Rules in, mp3 out.

## Two modes, one engine

```bash
python cli.py cue crystalline            # one-shot UI sound -> out/crystalline.{wav,mp3}
python cli.py cue all                    # every UI cue
python cli.py sculpture specs/unborn.json # a polymeter sculpture -> out/unborn.{wav,mp3}
```

- **cue** -- the app-facing SFX pipeline. Single gestures (chimes, prompts,
  thinking loops). Seeded from free-intelligence's RESONANCE voice cues; the
  ported cues render byte-identical to the originals.
- **sculpture** -- the Korda heart. Tracks with relatively prime loop lengths
  (7, 11, 13...) slip against each other into polymeter; modulator tracks reshape
  their targets each step (position = the phase slip, plus note / velocity /
  mute). The music is generated, not written.

## The spec is the glass box

A sculpture is a JSON document of tracks and modulations -- the seam an LLM
drives. Tell Claude *"make it darker, slower, add a track in 17"* and the spec
changes; the parameters stay human-readable. See [`specs/unborn.json`](specs/unborn.json).

```json
{
  "tracks": [
    { "name": "bell-7", "voice": "bell", "note": 60, "length": 7,
      "euclid": { "pulses": 3, "length": 7 } }
  ],
  "modulations": [ { "type": "position", "source": 3, "target": 0 } ]
}
```

## Roadmap

- **Phase 1 (this):** Python pipeline, rules -> mp3, spec-driven, LLM-authored.
- **Phase 2:** the DJ web surface -- Tone.js + Web Audio, live editable controls,
  the same spec grammar shared between Python and the browser (one grammar, two
  engines), an LLM tool-caller mutating the spec in real time.

## Requirements

`numpy`, `scipy`, `soundfile`, and `ffmpeg` on PATH (for mp3). See
`requirements.txt`.

## Credits

Algorithm and inspiration: **Chris Korda**, *Polymeter MIDI Sequencer*,
GPL-3.0 -- <https://github.com/victimofleisure/Polymeter>.

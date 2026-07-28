#!/usr/bin/env python3
"""unborn-songmaker CLI. Two modes, one engine.

  python cli.py cue crystalline                 -> out/crystalline.{wav,mp3}
  python cli.py cue all                          -> every UI cue
  python cli.py sculpture specs/unborn.json      -> render a polymeter sculpture

The sculpture mode is the Korda heart: rules generate the music. The cue mode is
the app-facing SFX pipeline. Both end in an mp3 ready to drop into a project."""
import argparse
import os
import sys

from unborn.cues import CUES, render_cue
from unborn.render import mix, to_mp3, write_wav
from unborn.spec import load

OUT = os.path.join(os.path.dirname(__file__), "out")


def emit(name: str, samples) -> None:
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, f"{name}.wav")
    write_wav(wav, samples)
    mp3 = to_mp3(wav)
    dur = len(samples) / 44100
    print(f"  {name}: {wav}{'  + .mp3' if mp3 else ''}  ({dur:.2f}s)")


def cmd_cue(name: str) -> None:
    names = list(CUES) if name == "all" else [name]
    for n in names:
        if n not in CUES:
            print(f"unknown cue '{n}'. known: {', '.join(CUES)}")
            sys.exit(1)
        emit(n, render_cue(n))


def cmd_sculpture(path: str) -> None:
    seq, spec = load(path)
    bars = spec.get("bars", 4)
    events = seq.run(bars=bars, beats_per_bar=spec.get("beats_per_bar", 4))
    print(f"  {len(events)} notes from {len(seq.tracks)} tracks, {len(seq.modulations)} modulations")
    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    emit(name, mix(events, sidechain=spec.get("sidechain"), rev=spec.get("reverb")))


def cmd_live(path: str, minutes: float | None, sink: str, dest: str | None) -> None:
    from unborn.live import LiveRenderer, pipe, play
    seq, spec = load(path)
    r = LiveRenderer(seq, spec)
    print(f"  {spec.get('name')}: {len(seq.tracks)} tracks, gain {r.calibrate():.3f}, "
          f"cycle {spec.get('bars')} bars — ctrl-c para parar")
    if sink == "speaker":
        play(r, minutes=minutes)
    else:
        os.makedirs(OUT, exist_ok=True)
        out = dest or os.path.join(OUT, f"{spec.get('name')}_live.wav")
        pipe(r, ["-y", out], minutes=minutes)
        print(f"  -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="unborn-songmaker")
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cue", help="render a one-shot UI sound cue")
    c.add_argument("name", help="cue name, or 'all'")
    s = sub.add_parser("sculpture", help="render a polymeter sculpture spec")
    s.add_argument("path", help="path to a sculpture .json spec")
    v = sub.add_parser("live", help="stream a sculpture forever (mode 2)")
    v.add_argument("path", help="path to a sculpture .json spec")
    v.add_argument("--minutes", type=float, default=None, help="stop after N minutes")
    v.add_argument("--sink", choices=["speaker", "ffmpeg"], default="speaker")
    v.add_argument("--dest", default=None, help="ffmpeg sink target (file or url)")
    args = p.parse_args()
    if args.mode == "cue":
        cmd_cue(args.name)
    elif args.mode == "sculpture":
        cmd_sculpture(args.path)
    else:
        cmd_live(args.path, args.minutes, args.sink, args.dest)


if __name__ == "__main__":
    main()

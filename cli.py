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
    emit(name, mix(events))


def main() -> None:
    p = argparse.ArgumentParser(prog="unborn-songmaker")
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cue", help="render a one-shot UI sound cue")
    c.add_argument("name", help="cue name, or 'all'")
    s = sub.add_parser("sculpture", help="render a polymeter sculpture spec")
    s.add_argument("path", help="path to a sculpture .json spec")
    args = p.parse_args()
    if args.mode == "cue":
        cmd_cue(args.name)
    else:
        cmd_sculpture(args.path)


if __name__ == "__main__":
    main()

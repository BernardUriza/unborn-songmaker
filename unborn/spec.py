"""Declarative spec loader: a JSON sculpture -> Tracks + Modulations. This is the
glass-box seam an LLM drives. Ask for a sound, Claude writes or edits the spec,
the engine renders it -- the parameters stay human-readable, never a black box."""
import json
import os

from .sequencer import Sequencer
from .track import (MOD_MUTE, MOD_NOTE, MOD_POSITION, MOD_VELOCITY, MODULATOR,
                    NOTE, Modulation, Track)

TOP_KEYS = {"name", "tempo", "ticks_per_beat", "bars", "beats_per_bar", "sidechain",
            "reverb", "tracks", "modulations"}
TRACK_KEYS = {"name", "type", "note", "length", "quant", "offset", "swing", "velocity",
              "voice", "steps", "euclid", "mute", "fx", "enter", "exit"}
FX_KEYS = {"reverse", "pitch", "granular", "ring", "crush", "downsample", "drive", "bandpass"}
MOD_TYPES = {MOD_MUTE, MOD_NOTE, MOD_POSITION, MOD_VELOCITY}
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


class SpecError(ValueError):
    """A spec that would render wrong in silence. Carries every problem at once so
    one edit pass fixes them all, instead of discovering them one crash at a time."""

    def __init__(self, errors: list[str], path: str | None = None):
        self.errors = errors
        head = f"{path}: " if path else ""
        super().__init__(head + f"{len(errors)} spec error(s)\n  - " + "\n  - ".join(errors))


def euclid(pulses: int, length: int, velocity: int = 96) -> list[int]:
    """Bjorklund euclidean rhythm: spread `pulses` as evenly as possible over
    `length` steps. A compact way to seed a track without hand-typing steps."""
    if length <= 0:
        return []
    pattern = []
    bucket = 0
    for _ in range(length):
        bucket += pulses
        if bucket >= length:
            bucket -= length
            pattern.append(velocity)
        else:
            pattern.append(0)
    return pattern


def _steps_from(spec: dict) -> list[int]:
    if "steps" in spec:
        return list(spec["steps"])
    if "euclid" in spec:
        e = spec["euclid"]
        return euclid(e["pulses"], e["length"], e.get("velocity", 96))
    return []


def validate(spec: dict, check_samples: bool = True) -> list[str]:
    """Every way a spec can silently mis-render, as a list (empty = valid). The
    engine is forgiving by design -- an unknown voice falls back to bell, a typo'd
    key is ignored -- so this is where a mistake gets a name instead of a sound."""
    from .render import ALL_VOICES
    errors: list[str] = []
    for k in set(spec) - TOP_KEYS:
        errors.append(f"unknown top-level key '{k}'")
    for k in ("tempo", "ticks_per_beat", "bars", "beats_per_bar"):
        if k in spec and not (isinstance(spec[k], (int, float)) and spec[k] > 0):
            errors.append(f"'{k}' must be > 0, got {spec[k]!r}")
    bars = spec.get("bars", 4)
    tracks = spec.get("tracks", [])
    if not tracks:
        errors.append("no tracks")
    for i, t in enumerate(tracks):
        where = f"track {i} '{t.get('name', '')}'"
        for k in set(t) - TRACK_KEYS:
            errors.append(f"{where}: unknown key '{k}'")
        kind = t.get("type", NOTE)
        if kind not in (NOTE, MODULATOR):
            errors.append(f"{where}: type must be '{NOTE}' or '{MODULATOR}', got '{kind}'")
        if "steps" in t and "euclid" in t:
            errors.append(f"{where}: has both 'steps' and 'euclid' -- pick one")
        steps = _steps_from(t)
        if not steps:
            errors.append(f"{where}: no steps")
        if any(not isinstance(v, int) for v in steps):
            errors.append(f"{where}: step values must be ints")
        elif kind == NOTE and any(not 0 <= v <= 127 for v in steps):
            errors.append(f"{where}: note-track step velocities must be in 0..127")
        quant = t.get("quant", 30)
        if not isinstance(quant, int) or quant < 1:
            errors.append(f"{where}: quant must be an int >= 1, got {quant!r}")
        elif abs(t.get("swing", 0)) >= quant:
            errors.append(f"{where}: |swing| {t.get('swing')} must be < quant {quant}")
        if kind == NOTE:
            voice = t.get("voice", "bell")
            if voice.startswith("sample:"):
                wav = os.path.join(SAMPLES_DIR, voice.split(":", 1)[1] + ".wav")
                if check_samples and not os.path.exists(wav):
                    errors.append(f"{where}: sample not found: {wav}")
            elif voice not in ALL_VOICES:
                errors.append(f"{where}: unknown voice '{voice}'")
        enter, exit_ = t.get("enter", 0), t.get("exit")
        if enter < 0 or enter >= bars:
            errors.append(f"{where}: enter {enter} outside 0..{bars - 1}")
        if exit_ is not None and exit_ <= enter:
            errors.append(f"{where}: exit {exit_} must be > enter {enter}")
        for k in set(t.get("fx") or {}) - FX_KEYS:
            errors.append(f"{where}: unknown fx '{k}'")
    for j, m in enumerate(spec.get("modulations", [])):
        where = f"modulation {j}"
        if m.get("type") not in MOD_TYPES:
            errors.append(f"{where}: type must be one of {sorted(MOD_TYPES)}, got {m.get('type')!r}")
        for end in ("source", "target"):
            ref = m.get(end)
            if not isinstance(ref, int) or not 0 <= ref < len(tracks):
                errors.append(f"{where}: {end} {ref!r} is not a track index")
        tgt = m.get("target")
        if isinstance(tgt, int) and 0 <= tgt < len(tracks) \
                and tracks[tgt].get("type", NOTE) == MODULATOR:
            errors.append(f"{where}: target is a modulator; recursive modulation is not supported")
    return errors


def track_from(spec: dict) -> Track:
    steps = _steps_from(spec)
    return Track(
        name=spec.get("name", ""),
        type=spec.get("type", "note"),
        note=spec.get("note", 64),
        length=spec.get("length", len(steps) or 16),
        quant=spec.get("quant", 30),
        offset=spec.get("offset", 0),
        swing=spec.get("swing", 0),
        velocity=spec.get("velocity", 0),
        voice=spec.get("voice", "bell"),
        steps=steps,
        mute=spec.get("mute", False),
        fx=spec.get("fx"),
        enter=spec.get("enter", 0),
        exit=spec.get("exit"),
    )


def sequencer_from(spec: dict) -> Sequencer:
    tracks = [track_from(t) for t in spec.get("tracks", [])]
    mods = [Modulation(m["type"], m["source"], m["target"])
            for m in spec.get("modulations", [])]
    return Sequencer(
        tracks=tracks,
        modulations=mods,
        tempo=spec.get("tempo", 120.0),
        ticks_per_beat=spec.get("ticks_per_beat", 24),
    )


def load(path: str) -> tuple[Sequencer, dict]:
    with open(path) as f:
        spec = json.load(f)
    errors = validate(spec)
    if errors:
        raise SpecError(errors, path)
    return sequencer_from(spec), spec

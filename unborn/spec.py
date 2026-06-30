"""Declarative spec loader: a JSON sculpture -> Tracks + Modulations. This is the
glass-box seam an LLM drives. Ask for a sound, Claude writes or edits the spec,
the engine renders it -- the parameters stay human-readable, never a black box."""
import json

from .sequencer import Sequencer
from .track import Modulation, Track


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
    return sequencer_from(spec), spec

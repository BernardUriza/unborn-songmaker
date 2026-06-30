"""One-shot UI sound cues -- the SFX side of the pipeline. Ported from
free-intelligence's RESONANCE cues so any app can ask for a sound and get an mp3.
These are single gestures, not sculptures: synthesized directly, no sequencer."""
import numpy as np

from .synth import additive, soft_attack, time_axis


def crystalline(dur: float = 0.7, f0: float = 880.0) -> np.ndarray:
    sig = additive(f0, dur, [(1.0, 1.0), (2.76, 0.55), (5.40, 0.28), (8.93, 0.13)])
    env = soft_attack(np.exp(-time_axis(dur) * 6.0))
    return sig * env * 0.42


def ready(note_dur: float = 0.14) -> np.ndarray:
    def note(freq: float) -> np.ndarray:
        t = time_axis(note_dur)
        sig = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
        env = np.sin(np.pi * t / note_dur) ** 1.5
        return sig * env
    return np.concatenate([note(523.25), note(783.99)]) * 0.3


def thinking(dur: float = 1.2, f: float = 174.61) -> np.ndarray:
    t = time_axis(dur)
    breaths = np.sin(np.pi * (t / dur) * 2.0) ** 2
    carrier = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t)
    return carrier * breaths * 0.12


CUES = {"crystalline": crystalline, "ready": ready, "thinking": thinking}


def render_cue(name: str) -> np.ndarray:
    return CUES[name]()

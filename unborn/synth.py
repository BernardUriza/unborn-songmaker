"""Additive synthesis primitives. Seeded from free-intelligence's
generate-resonance-cues.py: every voice is built from summed sine partials over a
shaped amplitude envelope, numpy only. No samples, no neural black box -- the
waveform is a rule you can read."""
import numpy as np

SR = 44100


def time_axis(dur: float) -> np.ndarray:
    return np.linspace(0.0, dur, max(1, int(SR * dur)), endpoint=False)


def midi_to_freq(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def soft_attack(env: np.ndarray, ms: float = 5.0) -> np.ndarray:
    n = max(1, int(SR * ms / 1000))
    env = env.copy()
    n = min(n, len(env))
    env[:n] *= np.linspace(0.0, 1.0, n)
    return env


def additive(freq: float, dur: float, partials: list[tuple[float, float]]) -> np.ndarray:
    t = time_axis(dur)
    sig = np.zeros_like(t)
    for ratio, amp in partials:
        sig += amp * np.sin(2 * np.pi * freq * ratio * t)
    return sig


BELL_PARTIALS = [(1.0, 1.0), (2.76, 0.55), (5.40, 0.28), (8.93, 0.13)]
HARMONIC_PARTIALS = [(1.0, 1.0), (2.0, 0.35), (3.0, 0.12)]


def bell_voice(freq: float, dur: float, decay: float = 6.0) -> np.ndarray:
    sig = additive(freq, dur, BELL_PARTIALS)
    env = soft_attack(np.exp(-time_axis(dur) * decay))
    return sig * env


def harmonic_voice(freq: float, dur: float, decay: float = 3.0) -> np.ndarray:
    sig = additive(freq, dur, HARMONIC_PARTIALS)
    env = soft_attack(np.exp(-time_axis(dur) * decay), ms=3.0)
    return sig * env


def pad_voice(freq: float, dur: float) -> np.ndarray:
    sig = additive(freq, dur, HARMONIC_PARTIALS)
    t = time_axis(dur)
    env = np.sin(np.pi * t / dur) ** 1.2
    return sig * env


def subbass_voice(freq: float, dur: float) -> np.ndarray:
    t = time_axis(dur)
    sig = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(2 * np.pi * freq * 2 * t)
    env = soft_attack(np.ones_like(t), ms=8.0)
    rel = min(len(env), int(SR * 0.05))
    env[-rel:] *= np.linspace(1.0, 0.0, rel)
    return np.tanh(sig * env * 1.2) * 0.7


VOICES = {
    "bell": bell_voice, "harmonic": harmonic_voice, "pad": pad_voice,
    "subbass": subbass_voice,
}


def render_voice(name: str, freq: float, dur: float) -> np.ndarray:
    return VOICES.get(name, bell_voice)(freq, dur)

"""Synthesized drum and bass voices -- the render layer. Korda emits MIDI to
external synths; this is the synth she leaves out, sitting strictly BELOW the
track/modulation contract. A kick/hat/bass is still just a Track with this voice.

Recipes follow the canonical techno formulas: kick = sine with a fast pitch
envelope (150->50 Hz) plus exponential gain decay; hat = bandpassed white noise
with a tiny decay; bass = sawtooth through a lowpass whose cutoff is enveloped."""
import numpy as np
from scipy import signal

from .synth import SR, time_axis


def _noise(n: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n)


def kick(freq: float = 50.0, dur: float = 0.34) -> np.ndarray:
    t = time_axis(dur)
    pitch = 50.0 + (150.0 - 50.0) * np.exp(-t / 0.03)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(phase) * np.exp(-t / 0.16)
    click = _noise(len(t)) * np.exp(-t / 0.004) * 0.3
    return np.tanh((body + click) * 1.4) * 0.9


def _hat(dur: float, tau: float) -> np.ndarray:
    t = time_axis(dur)
    sos = signal.butter(4, 7000, btype="highpass", fs=SR, output="sos")
    n = np.asarray(signal.sosfilt(sos, _noise(len(t))), dtype=np.float64)
    return n * np.exp(-t / tau) * 0.4


def hat(freq: float = 0.0, dur: float = 0.05) -> np.ndarray:
    return _hat(dur, 0.012)


def hat_open(freq: float = 0.0, dur: float = 0.18) -> np.ndarray:
    return _hat(dur, 0.07)


def clap(freq: float = 0.0, dur: float = 0.18) -> np.ndarray:
    t = time_axis(dur)
    sos = signal.butter(4, [1200, 3000], btype="bandpass", fs=SR, output="sos")
    n = np.asarray(signal.sosfilt(sos, _noise(len(t))), dtype=np.float64)
    env = np.exp(-t / 0.09)
    bursts = (np.sin(2 * np.pi * 90 * t) > 0.6)[: len(t)] * 0.5 + 0.5
    return n * env * bursts * 0.5


def bass(freq: float = 55.0, dur: float = 0.22) -> np.ndarray:
    t = time_axis(dur)
    saw = np.asarray(signal.sawtooth(2 * np.pi * freq * t), dtype=np.float64)
    cutoff = 200.0 + 1800.0 * np.exp(-t / 0.08)
    out = np.zeros_like(saw)
    block = 256
    for i in range(0, len(saw), block):
        fc = float(np.clip(cutoff[min(i, len(cutoff) - 1)], 60, SR / 2 - 100))
        sos = signal.butter(2, fc, btype="lowpass", fs=SR, output="sos")
        out[i:i + block] = signal.sosfilt(sos, saw[i:i + block])
    return out * np.exp(-t / 0.18) * 0.6


DRUM_VOICES = {
    "kick": kick, "hat": hat, "hat_open": hat_open, "clap": clap, "bass": bass,
}

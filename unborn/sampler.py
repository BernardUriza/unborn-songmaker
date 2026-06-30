"""Sampler voice -- the render layer for recorded audio. A Track with
voice "sample:<name>" triggers samples/<name>.wav, pitched by the track's note.
This is how Korda's spoken-vocal layer fits: it is still an ordinary Track in the
polymeter contract, so it inherits prime-length looping, swing and modulation."""
import os

import numpy as np
import soundfile as sf

from .synth import SR

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
BASE_NOTE = 60
_cache: dict[str, np.ndarray] = {}


def _load(name: str) -> np.ndarray:
    if name in _cache:
        return _cache[name]
    path = os.path.join(SAMPLES_DIR, f"{name}.wav")
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        new_len = int(len(data) * SR / sr)
        data = np.interp(np.linspace(0, len(data), new_len, endpoint=False),
                         np.arange(len(data)), data)
    _cache[name] = data
    return data


def render_sample(name: str, freq: float, dur: float) -> np.ndarray:
    data = _load(name)
    rate = freq / (440.0 * 2.0 ** ((BASE_NOTE - 69) / 12))
    if abs(rate - 1.0) > 1e-3 and rate > 0:
        idx = np.arange(0, len(data), rate)
        data = np.interp(idx, np.arange(len(data)), data)
    return data

"""Sampler voice -- the render layer for recorded audio. A Track with
voice "sample:<name>" triggers recorded audio, pitched by the track's note.
This is how Korda's spoken-vocal layer fits: it is still an ordinary Track in the
polymeter contract, so it inherits prime-length looping, swing and modulation.

`<name>` resolves to either
  - samples/<name>.wav       one recording (the spoken-voice layer), or
  - samples/<name>/*.wav     several TAKES of one gesture (home foley imported by
                             sfx_import.py) -- each trigger plays the next take,
                             round-robin, so a stamp on every bar never sounds
                             like one file looped.
The single file wins if both exist. Which take a trigger gets is counted by the
caller (render._bus, per voice, in event order), so a render stays deterministic."""
import glob
import os

import numpy as np
import soundfile as sf

from .synth import SR

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
BASE_NOTE = 60
_cache: dict[str, np.ndarray] = {}


def sample_takes(name: str, samples_dir: str | None = None) -> list[str]:
    """The wav files behind `sample:<name>`, in round-robin order ([] = missing)."""
    root = samples_dir or SAMPLES_DIR
    single = os.path.join(root, f"{name}.wav")
    if os.path.isfile(single):
        return [single]
    return sorted(glob.glob(os.path.join(root, name, "*.wav")))


def _load(path: str) -> np.ndarray:
    if path in _cache:
        return _cache[path]
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        new_len = int(len(data) * SR / sr)
        data = np.interp(np.linspace(0, len(data), new_len, endpoint=False),
                         np.arange(len(data)), data)
    _cache[path] = data
    return data


def render_sample(name: str, freq: float, dur: float, take: int = 0) -> np.ndarray:
    takes = sample_takes(name)
    if not takes:
        raise FileNotFoundError(f"sample not found: samples/{name}.wav or samples/{name}/*.wav")
    data = _load(takes[take % len(takes)])
    rate = freq / (440.0 * 2.0 ** ((BASE_NOTE - 69) / 12))
    if abs(rate - 1.0) > 1e-3 and rate > 0:
        idx = np.arange(0, len(data), rate)
        data = np.interp(idx, np.arange(len(data)), data)
    return data

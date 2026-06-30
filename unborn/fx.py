"""Per-voice effect chain -- render layer. A Track may carry an `fx` dict that is
applied to its rendered audio: ring modulation (metallic/robotic), bitcrush
(digital grit), downsample (lo-fi), drive (tanh saturation), bandpass (radio
voice). Built for the spoken-vocal layer -- Korda's vocals are processed, never
raw -- but any voice can use it."""
import numpy as np
from scipy import signal

from .synth import SR


def _granular(x: np.ndarray, grain_ms: float, scatter: float) -> np.ndarray:
    g = max(64, int(SR * grain_ms / 1000))
    hop = max(1, g // 2)
    win = np.hanning(g)
    rng = np.random.default_rng(0)
    out = np.zeros(len(x) + g, dtype=np.float64)
    pos = 0
    for start in range(0, len(x) - g, hop):
        jitter = int(rng.uniform(-scatter, scatter) * g)
        src = min(max(0, start + jitter), len(x) - g)
        out[pos:pos + g] += x[src:src + g] * win
        pos += hop
    return out[: len(x)]


def apply_fx(x: np.ndarray, fx: dict) -> np.ndarray:
    if not fx:
        return x
    x = np.asarray(x, dtype=np.float64)
    if fx.get("reverse"):
        x = x[::-1].copy()
    if "pitch" in fx:
        rate = 2.0 ** (float(fx["pitch"]) / 12.0)
        if rate > 0 and abs(rate - 1.0) > 1e-3:
            idx = np.arange(0, len(x), rate)
            x = np.interp(idx, np.arange(len(x)), x)
    if "granular" in fx:
        gp = fx["granular"]
        x = _granular(x, gp.get("grain_ms", 60), gp.get("scatter", 0.5))
    if "ring" in fx:
        t = np.arange(len(x)) / SR
        x = x * np.sin(2 * np.pi * float(fx["ring"]) * t)
    if "crush" in fx:
        levels = max(2, 2 ** int(fx["crush"]))
        x = np.round(x * levels) / levels
    if fx.get("downsample", 1) and int(fx.get("downsample", 1)) > 1:
        f = int(fx["downsample"])
        x = np.repeat(x[::f], f)[: len(x)]
    if "drive" in fx:
        d = 1.0 + float(fx["drive"]) * 12.0
        x = np.tanh(x * d) / np.tanh(d)
    if "bandpass" in fx:
        lo, hi = fx["bandpass"]
        sos = signal.butter(2, [float(lo), float(hi)], btype="bandpass", fs=SR, output="sos")
        x = np.asarray(signal.sosfilt(sos, x), dtype=np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * 0.95
    return x

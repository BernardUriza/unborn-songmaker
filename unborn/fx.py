"""Per-voice effect chain -- render layer. A Track may carry an `fx` dict that is
applied to its rendered audio: ring modulation (metallic/robotic), bitcrush
(digital grit), downsample (lo-fi), drive (tanh saturation), bandpass (radio
voice). Built for the spoken-vocal layer -- Korda's vocals are processed, never
raw -- but any voice can use it."""
import numpy as np
from scipy import signal

from .synth import SR


def apply_fx(x: np.ndarray, fx: dict) -> np.ndarray:
    if not fx:
        return x
    x = np.asarray(x, dtype=np.float64)
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

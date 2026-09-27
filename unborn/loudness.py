"""Loudness -- render layer. Peak normalisation (what `mix` did until now) makes a
sparse sculpture come out much quieter than a dense one: the peak is the same,
the perceived level is not. ITU-R BS.1770 measures what the ear hears; every
sculpture is brought to the same integrated loudness, then a look-ahead limiter
holds the true peak under the ceiling. Numpy + scipy only, no pyloudnorm."""
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

from .synth import SR

# BS.1770-4 K-weighting: a high shelf (head diffraction) then a high-pass (RLB).
# The reference coefficients are for 48 kHz; these design the same two filters
# at any rate (the standard's Annex 1 gives the target curves, not the fs).
_SHELF_F0, _SHELF_GAIN_DB, _SHELF_Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
_HPF_F0, _HPF_Q = 38.13547087602444, 0.5003270373238773


Biquad = tuple[np.ndarray, np.ndarray]


def _high_shelf(fs: float) -> Biquad:
    # De Man's bilinear form; at 48 kHz it reproduces the standard's Table 1 exactly.
    k = np.tan(np.pi * _SHELF_F0 / fs)
    vh = 10 ** (_SHELF_GAIN_DB / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / _SHELF_Q + k * k
    b = np.array([vh + vb * k / _SHELF_Q + k * k, 2 * (k * k - vh), vh - vb * k / _SHELF_Q + k * k]) / a0
    a = np.array([a0, 2 * (k * k - 1), 1 - k / _SHELF_Q + k * k]) / a0
    return b, a


def _high_pass(fs: float) -> Biquad:
    k = np.tan(np.pi * _HPF_F0 / fs)
    a0 = 1 + k / _HPF_Q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([a0, 2 * (k * k - 1), 1 - k / _HPF_Q + k * k]) / a0
    return b, a


def _as_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def k_weight(x: np.ndarray, fs: float = SR) -> np.ndarray:
    y = _as_stereo(x)
    for b, a in (_high_shelf(fs), _high_pass(fs)):
        y = np.asarray(signal.lfilter(b, a, y, axis=0), dtype=np.float64)
    return y


def integrated_lufs(x: np.ndarray, fs: float = SR) -> float:
    """BS.1770-4 integrated loudness: 400 ms blocks at 75 % overlap, absolute
    gate at -70 LUFS, relative gate 10 LU under the ungated mean. -inf for silence."""
    y = k_weight(x, fs)
    block, hop = int(0.4 * fs), int(0.1 * fs)
    if len(y) < block:
        y = np.pad(y, ((0, block - len(y)), (0, 0)))
    n_blocks = 1 + (len(y) - block) // hop
    idx = np.arange(block)[None, :] + hop * np.arange(n_blocks)[:, None]
    z = np.stack([np.mean(y[:, ch][idx] ** 2, axis=1) for ch in range(y.shape[1])], axis=1)
    power = z.sum(axis=1)  # channel weights G = 1 for L/R (no surround here)
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(power)
    keep = lk > -70.0
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(power[keep].mean()) - 10.0
    keep &= lk > rel
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10 * np.log10(power[keep].mean()))


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    """dBTP: sample peak after 4x oversampling, as BS.1770 Annex 2 prescribes."""
    y = signal.resample_poly(_as_stereo(x), oversample, 1, axis=0)
    peak = float(np.max(np.abs(y)))
    return 20 * np.log10(peak) if peak > 0 else float("-inf")


def limiter(x: np.ndarray, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
            release_ms: float = 60.0, fs: float = SR) -> np.ndarray:
    """Look-ahead brick-wall limiter on a shared stereo gain: the required gain
    is taken as a running minimum over the look-ahead window (so it reaches the
    floor before the peak arrives) and recovers with a one-pole release."""
    y = _as_stereo(x)
    # sample-peak limiting; inter-sample peaks overshoot by up to ~0.3 dB, so
    # hold the sample ceiling that much lower to keep the TRUE peak under `ceiling_db`
    ceiling = 10 ** ((ceiling_db - 0.3) / 20)
    peak = np.max(np.abs(y), axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    la = max(1, int(lookahead_ms * fs / 1000))
    held = minimum_filter1d(need, size=2 * la + 1, mode="nearest")
    coef = np.exp(-1.0 / (release_ms * fs / 1000))
    gain = np.empty_like(held)
    g = 1.0
    for i, h in enumerate(held):  # release: fall instantly, recover exponentially
        g = h if h < g else h + (g - h) * coef
        gain[i] = g
    return y * gain[:, None]


def normalize(x: np.ndarray, target_lufs: float = -14.0, ceiling_db: float = -1.0,
              fs: float = SR) -> tuple[np.ndarray, dict]:
    """Bring `x` to `target_lufs`, hold true peak at `ceiling_db`. Returns the
    audio and the receipts (measured before/after) so a caller can print them
    instead of trusting the intention."""
    y = _as_stereo(x)
    before = integrated_lufs(y, fs)
    gain = 10 ** ((target_lufs - before) / 20) if np.isfinite(before) else 1.0
    y = limiter(y * gain, ceiling_db, fs=fs)
    return y, {"lufs_in": before, "gain_db": 20 * np.log10(gain), "lufs": integrated_lufs(y, fs),
               "true_peak_db": true_peak_db(y)}

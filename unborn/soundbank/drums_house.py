import numpy as np
from scipy import signal
SR = 44100


def _n_samples(dur: float) -> int:
    return max(0, int(SR * max(0.0, float(dur))))


def _time(dur: float) -> np.ndarray:
    return np.arange(_n_samples(dur), dtype=np.float64) / SR


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float64)


def _fade(x: np.ndarray, ms: float = 3.0) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    if y.size == 0:
        return y
    fade_len = min(y.size // 2, max(1, int(SR * ms / 1000.0)))
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
    y[:fade_len] *= fade
    y[-fade_len:] *= fade[::-1]
    return y


def _normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    y = _fade(x)
    if y.size == 0:
        return y.astype(np.float64)
    max_abs = float(np.max(np.abs(y)))
    if max_abs > 1e-12:
        y = y * (peak / max_abs)
    return np.clip(y, -1.0, 1.0).astype(np.float64)


def _sweep_sine(start_hz: float, end_hz: float, t: np.ndarray, tau: float) -> np.ndarray:
    pitch = end_hz + (start_hz - end_hz) * np.exp(-t / tau)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    return np.sin(phase)


def _band_noise(n: int, seed: int, low: float | None = None, high: float | None = None) -> np.ndarray:
    src = _noise(n, seed)
    if n == 0:
        return src
    if low is not None and high is not None:
        sos = signal.butter(4, [low, high], btype="bandpass", fs=SR, output="sos")
    elif low is not None:
        sos = signal.butter(4, low, btype="highpass", fs=SR, output="sos")
    elif high is not None:
        sos = signal.butter(4, high, btype="lowpass", fs=SR, output="sos")
    else:
        return src
    return np.asarray(signal.sosfilt(sos, src), dtype=np.float64)


def _resonant_noise(n: int, seed: int, freq_hz: float, q: float = 16.0) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    b, a = signal.iirpeak(freq_hz, q, fs=SR)
    return np.asarray(signal.lfilter(b, a, _noise(n, seed)), dtype=np.float64)


def house_kick_punchy(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    body = _sweep_sine(175.0, 52.0, t, 0.026) * np.exp(-t / 0.18)
    click = _band_noise(t.size, 101, 1800.0, 5200.0) * np.exp(-t / 0.004) * 0.28
    thump = np.sin(2.0 * np.pi * 52.0 * t) * np.exp(-t / 0.24) * 0.42
    return _normalize(np.tanh((body + thump + click) * 1.55))


def house_kick_deep(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    body = _sweep_sine(125.0, 43.0, t, 0.045) * np.exp(-t / 0.32)
    sub = np.sin(2.0 * np.pi * 41.0 * t) * np.exp(-t / 0.46) * 0.55
    click = _band_noise(t.size, 102, 1000.0, 3600.0) * np.exp(-t / 0.006) * 0.12
    return _normalize(np.tanh((body + sub + click) * 1.35))


def house_kick_909(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    body = _sweep_sine(155.0, 48.0, t, 0.038) * np.exp(-t / 0.42)
    mid = np.sin(2.0 * np.pi * 96.0 * t + 0.45) * np.exp(-t / 0.055) * 0.18
    snap = _band_noise(t.size, 103, 2400.0, 7800.0) * np.exp(-t / 0.0035) * 0.18
    return _normalize(np.tanh((body + mid + snap) * 1.7))


def house_snare_tight(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    tone = np.sin(2.0 * np.pi * 188.0 * t) * np.exp(-t / 0.055) * 0.42
    noise = _band_noise(t.size, 201, 1350.0, 7200.0) * np.exp(-t / 0.045) * 0.75
    tick = _band_noise(t.size, 202, 5000.0, 11000.0) * np.exp(-t / 0.004) * 0.22
    return _normalize(tone + noise + tick)


def house_clap_layered(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    noise = _band_noise(t.size, 301, 950.0, 5300.0)
    burst_times = np.array([0.000, 0.014, 0.028, 0.043], dtype=np.float64)
    env = np.zeros_like(t)
    for bt in burst_times:
        env += np.where(t >= bt, np.exp(-(t - bt) / 0.010), 0.0)
    tail = np.where(t >= 0.036, np.exp(-(t - 0.036) / 0.115), 0.0) * 0.52
    low_smear = _band_noise(t.size, 302, 600.0, 1800.0) * np.exp(-t / 0.09) * 0.22
    return _normalize(noise * (env + tail) + low_smear)


def hat_closed_tight(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    noise = _band_noise(t.size, 401, 7200.0, 15000.0)
    metallic = _resonant_noise(t.size, 402, 9800.0, 24.0) * 0.28
    return _normalize((noise + metallic) * np.exp(-t / 0.018))


def hat_open_long(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    noise = _band_noise(t.size, 411, 6200.0, 14200.0)
    shimmer = _resonant_noise(t.size, 412, 11200.0, 18.0) * 0.35
    env = np.exp(-t / 0.23) * (1.0 - np.exp(-t / 0.006))
    return _normalize((noise + shimmer) * env)


def hat_offbeat_short(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    noise = _band_noise(t.size, 421, 5600.0, 12800.0)
    chirp = _resonant_noise(t.size, 422, 7600.0, 30.0) * 0.45
    return _normalize((noise * 0.8 + chirp) * np.exp(-t / 0.052))


def ride_metallic(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    partials = np.zeros_like(t)
    for ratio, amp in [(1.0, 0.55), (1.52, 0.30), (2.11, 0.22), (2.78, 0.16), (3.93, 0.12)]:
        partials += amp * np.sin(2.0 * np.pi * 360.0 * ratio * t)
    ping = partials * np.exp(-t / 0.62)
    air = _band_noise(t.size, 501, 6000.0, 16000.0) * np.exp(-t / 0.32) * 0.20
    return _normalize(ping + air)


def rim_click(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    stick = _resonant_noise(t.size, 601, 2450.0, 12.0) * np.exp(-t / 0.018)
    crack = _band_noise(t.size, 602, 1800.0, 9000.0) * np.exp(-t / 0.005) * 0.45
    low = np.sin(2.0 * np.pi * 540.0 * t) * np.exp(-t / 0.025) * 0.22
    return _normalize(stick + crack + low)


def perc_conga(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    pitch = 226.0 + 46.0 * np.exp(-t / 0.024)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(phase) * np.exp(-t / 0.18)
    slap = _band_noise(t.size, 701, 900.0, 4200.0) * np.exp(-t / 0.010) * 0.35
    return _normalize(body + slap)


def perc_bongo(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    pitch = 455.0 + 80.0 * np.exp(-t / 0.018)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(phase) * np.exp(-t / 0.105)
    tap = _band_noise(t.size, 711, 1600.0, 6200.0) * np.exp(-t / 0.007) * 0.28
    return _normalize(body + tap)


def perc_woodblock(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    hollow = _resonant_noise(t.size, 721, 1120.0, 18.0) * np.exp(-t / 0.055)
    top = _resonant_noise(t.size, 722, 2140.0, 22.0) * np.exp(-t / 0.034) * 0.58
    click = _band_noise(t.size, 723, 2800.0, 9000.0) * np.exp(-t / 0.003) * 0.20
    return _normalize(hollow + top + click)


def perc_shaker(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    noise = _band_noise(t.size, 731, 4800.0, 15500.0)
    pulse_rate = 72.0
    pulses = 0.35 + 0.65 * np.maximum(0.0, np.sin(2.0 * np.pi * pulse_rate * t))
    env = np.exp(-t / 0.12)
    return _normalize(noise * pulses * env)


def perc_tom_low(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    pitch = 92.0 + 45.0 * np.exp(-t / 0.040)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(phase) * np.exp(-t / 0.28)
    head = _band_noise(t.size, 801, 600.0, 2600.0) * np.exp(-t / 0.011) * 0.28
    return _normalize(body + head)


def perc_tom_mid(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    pitch = 148.0 + 62.0 * np.exp(-t / 0.034)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(phase) * np.exp(-t / 0.21)
    head = _band_noise(t.size, 811, 850.0, 3300.0) * np.exp(-t / 0.010) * 0.30
    return _normalize(body + head)


def crash_short(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    partials = np.zeros_like(t)
    for hz, amp in [(520.0, 0.32), (815.0, 0.25), (1240.0, 0.20), (1905.0, 0.15), (3060.0, 0.12)]:
        partials += amp * np.sin(2.0 * np.pi * hz * t + hz * 0.001)
    noise = _band_noise(t.size, 901, 3200.0, 17000.0)
    env = np.exp(-t / 0.36) * (1.0 - np.exp(-t / 0.005))
    return _normalize((noise * 0.82 + partials * 0.35) * env)


def zap_blip(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = float(freq) if freq > 0.0 else 180.0
    pitch = base + 1320.0 * np.exp(-t / 0.030)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    blip = signal.sawtooth(phase, width=0.35) * np.exp(-t / 0.085)
    bite = np.sin(phase * 1.93) * np.exp(-t / 0.024) * 0.25
    return _normalize(np.tanh((blip + bite) * 1.6))

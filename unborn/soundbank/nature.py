import numpy as np
from scipy import signal
SR = 44100


def _n(dur):
    return max(1, int(SR * max(0.0, float(dur))))


def _time(dur):
    return np.arange(_n(dur), dtype=np.float64) / SR


def _white(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, n).astype(np.float64)


def _pink(n, seed):
    x = _white(n, seed)
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float64)
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float64)
    return signal.lfilter(b, a, x).astype(np.float64)


def _butter(x, kind, freq, order=4):
    nyq = SR * 0.5
    if np.isscalar(freq):
        wn = min(0.999, max(0.001, float(freq) / nyq))
    else:
        lo = min(0.998, max(0.001, float(freq[0]) / nyq))
        hi = min(0.999, max(lo + 0.001, float(freq[1]) / nyq))
        wn = (lo, hi)
    b, a = signal.butter(order, wn, btype=kind)
    return signal.lfilter(b, a, x).astype(np.float64)


def _fade(x, ms=5.0):
    y = np.asarray(x, dtype=np.float64).copy()
    n = y.size
    k = min(n // 2, max(1, int(SR * ms / 1000.0)))
    ramp = np.linspace(0.0, 1.0, k, dtype=np.float64)
    y[:k] *= ramp
    y[-k:] *= ramp[::-1]
    return y


def _norm(x, peak=0.9):
    y = np.nan_to_num(np.asarray(x, dtype=np.float64), copy=True)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m > 1e-12:
        y *= peak / m
    return np.clip(y, -1.0, 1.0).astype(np.float64)


def _lfo(t, hz, phase=0.0):
    return 0.5 + 0.5 * np.sin(2.0 * np.pi * hz * t + phase)


def _impulse_train(t, rate, seed):
    rng = np.random.default_rng(seed)
    p = min(0.95, max(0.0, rate / SR))
    hits = (rng.random(t.size) < p).astype(np.float64)
    return hits * rng.uniform(0.35, 1.0, t.size)


def _resonant_bursts(t, rate, low, high, decay, seed):
    rng = np.random.default_rng(seed)
    n = t.size
    hits = _impulse_train(t, rate, seed + 1)
    idx = np.flatnonzero(hits > 0.0)
    out = np.zeros(n, dtype=np.float64)
    tail = max(8, int(decay * SR * 7.0))
    env = np.exp(-np.arange(tail, dtype=np.float64) / max(1.0, decay * SR))
    for i in idx:
        k = min(tail, n - i)
        f = rng.uniform(low, high)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        tt = np.arange(k, dtype=np.float64) / SR
        out[i:i + k] += hits[i] * env[:k] * np.sin(2.0 * np.pi * f * tt + ph)
    return out


def water_drip(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    hit = np.zeros(n, dtype=np.float64)
    centers = np.array([0.10, 0.34, 0.68, 0.87]) * max(t[-1] if n > 1 else 0.0, 0.1)
    for j, c in enumerate(centers):
        i = int(c * SR)
        k = min(n - i, int(SR * 0.16))
        if k > 0:
            tt = np.arange(k, dtype=np.float64) / SR
            env = np.exp(-tt * (28.0 + 8.0 * j))
            tone = np.sin(2.0 * np.pi * (740.0 + 110.0 * j) * tt + 20.0 * np.exp(-tt * 35.0))
            click = _butter(_white(k, 200 + j), "bandpass", (1800.0, 5600.0), 2) * np.exp(-tt * 90.0)
            hit[i:i + k] += 0.9 * env * tone + 0.22 * click
    return _norm(_fade(hit, 3.0))


def water_stream(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _butter(_pink(t.size, 301), "bandpass", (220.0, 3400.0), 3)
    sparkle = _butter(_white(t.size, 302), "highpass", 4200.0, 3) * 0.18
    ripple = 0.62 + 0.22 * _lfo(t, 1.7) + 0.12 * _lfo(t, 4.9, 1.4)
    bubbles = _resonant_bursts(t, 16.0, 260.0, 920.0, 0.018, 303) * 0.16
    return _norm(_fade((base + sparkle) * ripple + bubbles, 8.0))


def water_bubble(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    burble = _resonant_bursts(t, 8.0, 130.0, 420.0, 0.055, 401)
    air = _butter(_white(t.size, 402), "bandpass", (700.0, 2200.0), 2) * (0.08 + 0.5 * np.maximum(0.0, burble))
    wobble = 0.8 + 0.2 * np.sin(2.0 * np.pi * 6.0 * t + 1.0)
    return _norm(_fade(burble * wobble + air, 5.0))


def ocean_wave(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    surf = _butter(_pink(t.size, 501), "lowpass", 1500.0, 4)
    foam = _butter(_white(t.size, 502), "bandpass", (900.0, 5200.0), 3) * 0.45
    swell = 0.18 + 0.82 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.11 * t - 1.2)) ** 2
    wash = 0.55 + 0.20 * _lfo(t, 0.41, 0.7) + 0.08 * _lfo(t, 1.3, 2.2)
    return _norm(_fade((surf + foam * swell) * swell * wash, 20.0))


def rain_soft(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    bed = _butter(_white(t.size, 601), "bandpass", (1400.0, 7600.0), 3) * 0.22
    drops = _resonant_bursts(t, 70.0, 1600.0, 6200.0, 0.004, 602) * 0.42
    veil = _butter(_pink(t.size, 603), "highpass", 2400.0, 2) * 0.12
    mod = 0.74 + 0.18 * _lfo(t, 0.55, 1.2)
    return _norm(_fade((bed + drops + veil) * mod, 6.0))


def wind_gust(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    low = _butter(_pink(t.size, 701), "bandpass", (90.0, 1300.0), 4)
    breath = _butter(_white(t.size, 702), "bandpass", (700.0, 3100.0), 2) * 0.24
    gust = (0.16 + 0.84 * _lfo(t, 0.18, -1.1) ** 3) * (0.75 + 0.25 * _lfo(t, 0.9, 0.5))
    return _norm(_fade((low + breath) * gust, 18.0))


def wind_howl(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    noise = _butter(_pink(t.size, 801), "bandpass", (180.0, 1800.0), 4)
    f = 360.0 + 120.0 * np.sin(2.0 * np.pi * 0.16 * t) + 55.0 * np.sin(2.0 * np.pi * 0.47 * t + 1.9)
    phase = np.cumsum(f) * (2.0 * np.pi / SR)
    tone = np.sin(phase) + 0.35 * np.sin(phase * 1.97)
    env = 0.24 + 0.76 * _lfo(t, 0.24, -0.5) ** 2
    return _norm(_fade(noise * env + tone * env * 0.22, 12.0))


def leaves_rustle(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    dry = _butter(_white(t.size, 901), "bandpass", (1800.0, 9000.0), 3)
    taps = _impulse_train(t, 45.0, 902)
    kernel = np.exp(-np.arange(max(8, int(SR * 0.035)), dtype=np.float64) / (SR * 0.006))
    flicker = signal.lfilter(kernel, [1.0], taps)
    wave = (0.12 + 0.95 * np.minimum(1.0, flicker)) * (0.65 + 0.25 * _lfo(t, 2.8))
    return _norm(_fade(dry * wave + _butter(_pink(t.size, 903), "highpass", 3500.0, 2) * 0.08, 5.0))


def fire_crackle(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    hiss = _butter(_white(t.size, 1001), "bandpass", (900.0, 7200.0), 3) * 0.17
    pops = _resonant_bursts(t, 18.0, 900.0, 5200.0, 0.009, 1002)
    low = _butter(_pink(t.size, 1003), "lowpass", 500.0, 2) * (0.22 + 0.12 * _lfo(t, 3.2))
    return _norm(_fade(hiss + pops * 0.9 + low, 4.0))


def lava_bubble(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    thick = _butter(_pink(t.size, 1101), "lowpass", 420.0, 4) * (0.45 + 0.22 * _lfo(t, 0.75))
    globs = _resonant_bursts(t, 5.0, 55.0, 210.0, 0.12, 1102)
    sizzle = _butter(_white(t.size, 1103), "bandpass", (1300.0, 4200.0), 2) * (0.06 + 0.12 * np.abs(globs))
    return _norm(_fade(thick + globs * 1.2 + sizzle, 10.0))


def thunder_rumble(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    low = _butter(_pink(t.size, 1201), "lowpass", 170.0, 4)
    sub = np.sin(2.0 * np.pi * (42.0 + 9.0 * _lfo(t, 0.29)) * t)
    strike = np.exp(-t * 0.7) * (1.0 - np.exp(-t * 18.0))
    roll = 0.42 + 0.32 * _lfo(t, 0.43, 1.4) + 0.16 * _lfo(t, 1.1)
    distant = _butter(_white(t.size, 1202), "bandpass", (220.0, 850.0), 3) * np.exp(-t * 0.35) * 0.18
    return _norm(_fade((low * roll + sub * 0.38) * strike + distant, 8.0))


def ember_hiss(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    hiss = _butter(_white(t.size, 1301), "highpass", 5200.0, 3)
    glow = _butter(_pink(t.size, 1302), "bandpass", (500.0, 1800.0), 2) * 0.10
    pulse = 0.38 + 0.16 * _lfo(t, 0.7, 0.8) + 0.08 * _lfo(t, 4.1)
    sparks = _resonant_bursts(t, 8.0, 3500.0, 9000.0, 0.003, 1303) * 0.25
    return _norm(_fade(hiss * pulse + glow + sparks, 5.0))


def splash(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    impact = _butter(_white(t.size, 1401), "bandpass", (180.0, 2600.0), 4) * np.exp(-t * 8.0)
    spray = _butter(_white(t.size, 1402), "highpass", 2600.0, 3) * np.exp(-t * 12.0)
    droplets = _resonant_bursts(t, 28.0, 650.0, 4200.0, 0.018, 1403) * np.exp(-t * 2.4)
    body = np.sin(2.0 * np.pi * 115.0 * t + 10.0 * np.exp(-t * 18.0)) * np.exp(-t * 9.0)
    return _norm(_fade(impact * 0.8 + spray * 0.34 + droplets * 0.65 + body * 0.35, 3.0))


def geyser(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    roar = _butter(_pink(t.size, 1501), "bandpass", (120.0, 2600.0), 4)
    steam = _butter(_white(t.size, 1502), "highpass", 3600.0, 3) * 0.36
    rise = 1.0 - np.exp(-t * 3.0)
    churn = 0.64 + 0.22 * _lfo(t, 2.2) + 0.08 * _lfo(t, 8.3, 0.4)
    spurts = _resonant_bursts(t, 10.0, 160.0, 760.0, 0.04, 1503) * 0.33
    return _norm(_fade((roar + steam) * rise * churn + spurts, 12.0))


def steam_hiss(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    hiss = _butter(_white(t.size, 1601), "bandpass", (3600.0, 11800.0), 4)
    pressure = 0.72 + 0.18 * _lfo(t, 1.6, 0.2) + 0.07 * _lfo(t, 7.0)
    pipe = _butter(_pink(t.size, 1602), "bandpass", (900.0, 2200.0), 2) * 0.08
    return _norm(_fade(hiss * pressure + pipe, 5.0))


def gravel_crunch(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    grit = _butter(_white(t.size, 1701), "bandpass", (350.0, 6500.0), 3)
    steps = _impulse_train(t, 24.0, 1702)
    kernel = np.exp(-np.arange(max(8, int(SR * 0.09)), dtype=np.float64) / (SR * 0.018))
    crush = signal.lfilter(kernel, [1.0], steps)
    knocks = _resonant_bursts(t, 14.0, 180.0, 900.0, 0.026, 1703)
    return _norm(_fade(grit * np.minimum(1.0, crush * 1.8) + knocks * 0.45, 4.0))


def forest_air(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    air = _butter(_pink(t.size, 1801), "bandpass", (120.0, 2200.0), 4)
    canopy = _butter(_white(t.size, 1802), "bandpass", (2200.0, 7200.0), 2) * 0.10
    breeze = 0.42 + 0.28 * _lfo(t, 0.12, -0.8) + 0.10 * _lfo(t, 0.9)
    distant = 0.035 * np.sin(2.0 * np.pi * 880.0 * t + 0.7 * np.sin(2.0 * np.pi * 4.8 * t))
    distant *= 0.5 + 0.5 * _lfo(t, 0.07, 2.1)
    return _norm(_fade((air + canopy) * breeze + distant, 15.0))

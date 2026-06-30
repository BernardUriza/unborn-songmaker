import numpy as np
from scipy import signal
SR = 44100


def _n_samples(dur: float) -> int:
    return max(0, int(SR * max(0.0, float(dur))))


def _time(dur: float) -> np.ndarray:
    return np.arange(_n_samples(dur), dtype=np.float64) / SR


def _base(freq: float, default: float) -> float:
    f = float(freq)
    return f if f > 0.0 else default


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float64)


def _fade(x: np.ndarray, ms: float = 4.0) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    if y.size == 0:
        return y
    fade_len = min(y.size // 2, max(1, int(SR * ms / 1000.0)))
    if fade_len > 0:
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
        y *= peak / max_abs
    return np.clip(y, -1.0, 1.0).astype(np.float64)


def _phase(hz: np.ndarray | float) -> np.ndarray:
    return 2.0 * np.pi * np.cumsum(hz) / SR


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


def _resonant_noise(n: int, seed: int, hz: float, q: float) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    b, a = signal.iirpeak(hz, q, fs=SR)
    return np.asarray(signal.lfilter(b, a, _noise(n, seed)), dtype=np.float64)


def _adsr_like(t: np.ndarray, attack: float, release: float) -> np.ndarray:
    if t.size == 0:
        return t
    end = t[-1] if t[-1] > 0.0 else 1.0 / SR
    a = np.minimum(1.0, t / max(attack, 1.0 / SR))
    r = np.minimum(1.0, np.maximum(0.0, (end - t) / max(release, 1.0 / SR)))
    return np.minimum(a, r)


def car_pass(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    base = _base(freq, 92.0)
    pos = t / max(float(dur), 1.0 / SR)
    doppler = base * (1.42 - 0.68 / (1.0 + np.exp(-(pos - 0.52) * 16.0)))
    engine = signal.sawtooth(_phase(doppler), width=0.58) * 0.28
    tires = _band_noise(n, 1101, 80.0, 950.0)
    whoosh = _band_noise(n, 1102, 350.0, 3400.0)
    pass_env = np.exp(-((pos - 0.50) / 0.30) ** 2)
    near_side = 0.65 + 0.35 * np.tanh((pos - 0.48) * 12.0)
    return _normalize((tires * 0.55 + whoosh * 0.25 + engine * near_side) * pass_env)


def car_horn(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 410.0)
    vibrato = 5.5 * np.sin(2.0 * np.pi * 5.2 * t)
    phase1 = _phase(base + vibrato)
    phase2 = _phase(base * 1.19 + vibrato * 0.8)
    tone = np.sin(phase1) * 0.72 + np.sin(phase2) * 0.42 + np.sin(phase1 * 2.0) * 0.12
    env = _adsr_like(t, 0.025, 0.045) * (0.88 + 0.12 * np.sin(2.0 * np.pi * 7.0 * t))
    return _normalize(np.tanh(tone * env * 1.4))


def ambulance_siren(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 660.0)
    alt = 0.5 + 0.5 * signal.square(2.0 * np.pi * 1.75 * t, duty=0.50)
    target = base * (1.0 + alt * 0.34)
    glide = signal.sosfilt(signal.butter(2, 7.0, btype="lowpass", fs=SR, output="sos"), target)
    vibrato = 8.0 * np.sin(2.0 * np.pi * 6.2 * t)
    phase = _phase(glide + vibrato)
    tone = np.sin(phase) + 0.25 * np.sin(phase * 2.0)
    return _normalize(tone * _adsr_like(t, 0.020, 0.035))


def police_siren(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 520.0)
    sweep = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.82 * t)
    pitch = base + base * 0.62 * sweep + 11.0 * np.sin(2.0 * np.pi * 5.6 * t)
    phase = _phase(pitch)
    grit = _band_noise(t.size, 1201, 900.0, 2800.0) * 0.045
    tone = np.sin(phase) * 0.92 + np.sin(phase * 1.01 + 0.4) * 0.22 + grit
    return _normalize(tone * _adsr_like(t, 0.018, 0.030))


def traffic_hum(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 58.0)
    n = t.size
    low = _band_noise(n, 1301, 25.0, 210.0) * 0.75
    road = _band_noise(n, 1302, 180.0, 1500.0) * 0.28
    engines = np.zeros_like(t)
    for ratio, amp, wobble in [(1.0, 0.28, 0.19), (1.31, 0.18, 0.27), (1.87, 0.12, 0.13)]:
        hz = base * ratio * (1.0 + 0.025 * np.sin(2.0 * np.pi * wobble * t))
        engines += amp * signal.sawtooth(_phase(hz), width=0.62)
    env = 0.82 + 0.18 * np.sin(2.0 * np.pi * 0.11 * t + 0.6)
    return _normalize((low + road + engines) * env)


def subway_rumble(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    base = _base(freq, 43.0)
    rail_rate = 9.0 + 1.7 * np.sin(2.0 * np.pi * 0.22 * t)
    pulses = 0.48 + 0.52 * np.maximum(0.0, signal.square(_phase(rail_rate), duty=0.35))
    sub = np.sin(_phase(base * (1.0 + 0.04 * np.sin(2.0 * np.pi * 0.35 * t)))) * 0.45
    rumble = _band_noise(n, 1401, 18.0, 170.0) * pulses
    metal = _resonant_noise(n, 1402, 780.0, 18.0) * 0.10
    squeal = np.sin(_phase(1850.0 + 260.0 * np.sin(2.0 * np.pi * 0.18 * t))) * 0.06
    return _normalize((rumble * 0.82 + sub + metal + squeal) * _adsr_like(t, 0.080, 0.080))


def crowd_murmur(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    base = _base(freq, 145.0)
    murmur = np.zeros_like(t)
    formants = [(420.0, 9.0), (770.0, 11.0), (1180.0, 13.0), (2300.0, 10.0)]
    for i, (hz, q) in enumerate(formants):
        band = _resonant_noise(n, 1501 + i, hz, q)
        lfo = 0.55 + 0.45 * np.sin(2.0 * np.pi * (0.23 + i * 0.11) * t + i)
        murmur += band * lfo * (0.38 / (i + 1.0))
    voices = np.zeros_like(t)
    for i, ratio in enumerate([0.82, 1.0, 1.13, 1.41, 1.72]):
        hz = base * ratio * (1.0 + 0.018 * np.sin(2.0 * np.pi * (0.31 + i * 0.07) * t))
        voices += np.sin(_phase(hz)) * (0.05 + i * 0.006)
    breath = _band_noise(n, 1510, 180.0, 2800.0) * 0.18
    return _normalize((murmur + voices + breath) * _adsr_like(t, 0.060, 0.070))


def footsteps(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    step_rate = max(1.2, _base(freq, 2.0))
    out = np.zeros_like(t)
    step_times = np.arange(0.06, max(float(dur), 0.0) + 0.001, 1.0 / step_rate)
    gravel = _band_noise(n, 1601, 700.0, 3600.0)
    for i, st in enumerate(step_times):
        dt = t - st
        hit = dt >= 0.0
        thud = np.sin(2.0 * np.pi * (82.0 + 12.0 * (i % 2)) * dt) * np.exp(-dt / 0.040) * hit
        scrape = gravel * np.exp(-dt / 0.018) * hit
        out += thud * 0.62 + scrape * (0.18 + 0.06 * (i % 2))
    return _normalize(out)


def door_slam(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    base = _base(freq, 86.0)
    impact = _band_noise(n, 1701, 45.0, 1600.0) * np.exp(-t / 0.035)
    crack = _band_noise(n, 1702, 1200.0, 6200.0) * np.exp(-t / 0.006) * 0.42
    panel = np.sin(_phase(base * (1.0 + 0.18 * np.exp(-t / 0.050)))) * np.exp(-t / 0.28)
    rattle = _resonant_noise(n, 1703, 310.0, 14.0) * np.exp(-t / 0.12) * 0.45
    return _normalize(np.tanh((impact + crack + panel * 0.75 + rattle) * 1.65))


def bell_church(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 196.0)
    bell = np.zeros_like(t)
    partials = [(1.00, 1.00, 3.8), (1.48, 0.58, 3.1), (2.01, 0.38, 2.5),
                (2.74, 0.27, 1.9), (3.76, 0.18, 1.4), (5.03, 0.12, 1.0)]
    for i, (ratio, amp, decay) in enumerate(partials):
        detune = 1.0 + 0.002 * np.sin(2.0 * np.pi * (0.07 + i * 0.03) * t)
        bell += amp * np.sin(_phase(base * ratio * detune) + i * 0.4) * np.exp(-t / decay)
    strike = _band_noise(t.size, 1801, 900.0, 7000.0) * np.exp(-t / 0.018) * 0.16
    return _normalize((bell + strike) * (1.0 - np.exp(-t / 0.010)))


def clock_tick(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    out = np.zeros_like(t)
    interval = 1.0 / max(0.5, _base(freq, 1.0))
    ticks = np.arange(0.0, max(float(dur), 0.0) + 0.001, interval)
    click_noise = _band_noise(n, 1901, 2800.0, 12000.0)
    for i, tick in enumerate(ticks):
        dt = t - tick
        hit = dt >= 0.0
        pitch = 2200.0 if i % 2 == 0 else 1650.0
        click = np.sin(2.0 * np.pi * pitch * dt) * np.exp(-dt / 0.010) * hit
        snap = click_noise * np.exp(-dt / 0.0025) * hit
        out += click * 0.58 + snap * 0.20
    return _normalize(out)


def glass_break(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    crash = _band_noise(n, 2001, 1400.0, 14000.0) * np.exp(-t / 0.16)
    shards = np.zeros_like(t)
    for i, hz in enumerate([1740.0, 2380.0, 3150.0, 4680.0, 6120.0, 9040.0]):
        delay = 0.006 + i * 0.011
        dt = np.maximum(0.0, t - delay)
        shards += np.sin(2.0 * np.pi * hz * dt + i) * np.exp(-dt / (0.09 + i * 0.015)) * (t >= delay) * (0.30 / (i + 1.0))
    initial = _band_noise(n, 2002, 350.0, 9000.0) * np.exp(-t / 0.010) * 0.72
    return _normalize(np.tanh((initial + crash * 0.75 + shards) * 1.7))


def elevator_ding(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 880.0)
    first = np.sin(_phase(base)) * np.exp(-t / 0.85)
    delay = 0.115
    dt = np.maximum(0.0, t - delay)
    second = np.sin(2.0 * np.pi * base * 1.25 * dt) * np.exp(-dt / 0.72) * (t >= delay)
    shimmer = np.sin(_phase(base * 2.5)) * np.exp(-t / 0.22) * 0.10
    return _normalize((first * 0.72 + second * 0.54 + shimmer) * (1.0 - np.exp(-t / 0.006)))


def phone_ring(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 440.0)
    cadence = (signal.square(2.0 * np.pi * 1.25 * t, duty=0.46) > 0.0).astype(np.float64)
    edge = signal.sosfilt(signal.butter(2, 18.0, btype="lowpass", fs=SR, output="sos"), cadence)
    warble = 11.0 * np.sin(2.0 * np.pi * 18.0 * t)
    tone = np.sin(_phase(base + warble)) * 0.68 + np.sin(_phase(base * 1.25 - warble * 0.4)) * 0.48
    buzz = signal.square(_phase(base * 2.0 + warble), duty=0.52) * 0.08
    return _normalize((tone + buzz) * edge * _adsr_like(t, 0.010, 0.025))


def helicopter(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    base = _base(freq, 54.0)
    rotor_rate = 10.5
    blades = 0.30 + 0.70 * (0.5 + 0.5 * signal.square(2.0 * np.pi * rotor_rate * t, duty=0.24))
    blade_env = signal.sosfilt(signal.butter(2, 42.0, btype="lowpass", fs=SR, output="sos"), blades)
    low = np.sin(_phase(base * (1.0 + 0.04 * np.sin(2.0 * np.pi * 0.65 * t)))) * blade_env
    chop = _band_noise(n, 2101, 90.0, 1100.0) * blade_env * 0.52
    air = _band_noise(n, 2102, 900.0, 3600.0) * (0.16 + 0.24 * blade_env)
    return _normalize((low * 0.86 + chop + air) * _adsr_like(t, 0.040, 0.060))


def construction_drill(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    n = t.size
    base = _base(freq, 92.0)
    chatter_rate = 28.0 + 4.0 * np.sin(2.0 * np.pi * 0.7 * t)
    chatter = 0.45 + 0.55 * (0.5 + 0.5 * signal.square(_phase(chatter_rate), duty=0.38))
    motor = signal.sawtooth(_phase(base * (1.0 + 0.05 * np.sin(2.0 * np.pi * 6.0 * t))), width=0.44)
    bit = _band_noise(n, 2201, 900.0, 5200.0) * chatter
    metal = _resonant_noise(n, 2202, 2850.0, 26.0) * chatter * 0.35
    return _normalize(np.tanh((motor * 0.54 + bit * 0.68 + metal) * _adsr_like(t, 0.025, 0.040) * 1.9))


def train_horn(freq: float, dur: float) -> np.ndarray:
    t = _time(dur)
    base = _base(freq, 185.0)
    bend = 1.0 - 0.035 * np.exp(-t / 0.20)
    chord = np.zeros_like(t)
    for ratio, amp in [(1.0, 0.82), (1.20, 0.55), (1.50, 0.34), (2.01, 0.18)]:
        phase = _phase(base * ratio * bend + 2.5 * np.sin(2.0 * np.pi * 4.8 * t))
        chord += amp * np.sin(phase)
    breath = _band_noise(t.size, 2301, 120.0, 1200.0) * 0.16
    env = _adsr_like(t, 0.090, 0.120) * (0.94 + 0.06 * np.sin(2.0 * np.pi * 0.8 * t))
    return _normalize(np.tanh((chord + breath) * env * 1.35))

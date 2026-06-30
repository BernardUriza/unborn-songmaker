import numpy as np
from scipy import signal
SR = 44100


def _n_samples(dur: float) -> int:
    return max(0, int(SR * max(0.0, float(dur))))


def _time(dur: float) -> np.ndarray:
    return np.arange(_n_samples(dur), dtype=np.float64) / SR


def _freq(freq: float) -> float:
    if not np.isfinite(freq):
        return 440.0
    return float(np.clip(freq, 20.0, 12000.0))


def _phase(freq_hz: float, t: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * freq_hz * t


def _fade(x: np.ndarray, ms: float = 4.0) -> np.ndarray:
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


def _exp_env(t: np.ndarray, decay: float, attack: float = 0.003) -> np.ndarray:
    if t.size == 0:
        return t
    return (1.0 - np.exp(-t / max(attack, 1e-5))) * np.exp(-t / max(decay, 1e-5))


def _asr_env(t: np.ndarray, attack: float, release: float) -> np.ndarray:
    if t.size == 0:
        return t
    dur = (t.size - 1) / SR
    env = np.ones_like(t)
    env *= np.minimum(1.0, t / max(attack, 1e-5))
    env *= np.minimum(1.0, np.maximum(0.0, dur - t) / max(release, 1e-5))
    return env


def _adsr_env(t: np.ndarray, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    if t.size == 0:
        return t
    dur = (t.size - 1) / SR
    a = max(attack, 1e-5)
    d = max(decay, 1e-5)
    r = max(release, 1e-5)
    env = np.ones_like(t) * sustain
    attack_mask = t < a
    decay_mask = (t >= a) & (t < a + d)
    env[attack_mask] = t[attack_mask] / a
    env[decay_mask] = 1.0 - (1.0 - sustain) * ((t[decay_mask] - a) / d)
    release_start = max(0.0, dur - r)
    release_mask = t >= release_start
    if np.any(release_mask):
        env[release_mask] *= np.maximum(0.0, (dur - t[release_mask]) / r)
    return np.clip(env, 0.0, 1.0)


def _seed(freq_hz: float, salt: int) -> int:
    return int((abs(freq_hz) * 1000.0 + salt * 7919.0) % (2**32 - 1))


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, n).astype(np.float64)


def _lowpass(x: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    cutoff = float(np.clip(cutoff, 30.0, SR * 0.48))
    sos = signal.butter(order, cutoff, btype="lowpass", fs=SR, output="sos")
    return np.asarray(signal.sosfilt(sos, x), dtype=np.float64)


def _highpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    cutoff = float(np.clip(cutoff, 20.0, SR * 0.45))
    sos = signal.butter(order, cutoff, btype="highpass", fs=SR, output="sos")
    return np.asarray(signal.sosfilt(sos, x), dtype=np.float64)


def _bandpass(x: np.ndarray, low: float, high: float, order: int = 2) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    low = float(np.clip(low, 20.0, SR * 0.45))
    high = float(np.clip(high, low + 10.0, SR * 0.48))
    sos = signal.butter(order, [low, high], btype="bandpass", fs=SR, output="sos")
    return np.asarray(signal.sosfilt(sos, x), dtype=np.float64)


def _karplus(freq_hz: float, n: int, decay: float, brightness: float, seed: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    delay = max(2, int(SR / max(freq_hz, 20.0)))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1.0, 1.0, delay).astype(np.float64)
    if brightness < 1.0:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        buf = brightness * buf + (1.0 - brightness) * np.convolve(buf, kernel, mode="same")
    out = np.zeros(n, dtype=np.float64)
    idx = 0
    for i in range(n):
        current = buf[idx]
        nxt = buf[(idx + 1) % delay]
        out[i] = current
        buf[idx] = decay * (0.5 * current + 0.5 * nxt)
        idx = (idx + 1) % delay
    return out


def _additive(t: np.ndarray, freq_hz: float, partials: list[tuple[float, float, float]]) -> np.ndarray:
    y = np.zeros_like(t)
    for ratio, amp, tau in partials:
        y += amp * np.sin(_phase(freq_hz * ratio, t)) * np.exp(-t / max(tau, 1e-5))
    return y


def pluck_guitar_nylon(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    string = _karplus(f, t.size, 0.992, 0.42, _seed(f, 11))
    body = np.sin(_phase(f, t)) * np.exp(-t / 0.36) * 0.18
    warm = _lowpass(string, min(6200.0, f * 14.0))
    return _normalize(warm * np.exp(-t / 1.25) + body)


def pluck_guitar_steel(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    string = _karplus(f, t.size, 0.996, 0.86, _seed(f, 12))
    shimmer = _highpass(string, max(850.0, f * 2.5)) * 0.23
    return _normalize(_lowpass(string + shimmer, 11200.0) * np.exp(-t / 1.9))


def pluck_bass_finger(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    string = _karplus(f, t.size, 0.997, 0.32, _seed(f, 13))
    sub = np.sin(_phase(f * 0.5, t)) * np.exp(-t / 0.9) * 0.34
    attack = _bandpass(_noise(t.size, _seed(f, 14)), 450.0, 1800.0) * np.exp(-t / 0.012) * 0.20
    return _normalize(_lowpass(string, 2600.0) * np.exp(-t / 1.55) + sub + attack)


def piano_soft(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    partials = [(1.0, 1.0, 0.95), (2.0, 0.42, 0.58), (3.01, 0.18, 0.36), (4.02, 0.08, 0.22)]
    hammer = _bandpass(_noise(t.size, _seed(f, 21)), 900.0, 3600.0) * np.exp(-t / 0.010) * 0.06
    tone = _additive(t, f, partials) * _exp_env(t, 1.2, 0.006)
    return _normalize(_lowpass(tone + hammer, 6800.0))


def piano_bright(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    partials = [(1.0, 0.95, 0.85), (2.01, 0.62, 0.55), (3.03, 0.35, 0.34), (4.06, 0.20, 0.24), (5.09, 0.11, 0.16)]
    hammer = _bandpass(_noise(t.size, _seed(f, 22)), 1800.0, 7600.0) * np.exp(-t / 0.007) * 0.12
    tone = _additive(t, f, partials) * _exp_env(t, 0.95, 0.003)
    return _normalize(_lowpass(tone + hammer, 11500.0))


def ep_rhodes(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    bell_env = np.exp(-t / 0.28)
    mod = np.sin(_phase(f * 2.01, t)) * (2.8 * bell_env)
    bell = np.sin(_phase(f, t) + mod) * bell_env * 0.65
    tine = np.sin(_phase(f * 3.0, t + 0.0007)) * np.exp(-t / 0.22) * 0.18
    body = np.sin(_phase(f * 0.5, t)) * _adsr_env(t, 0.012, 0.18, 0.45, 0.18) * 0.55
    trem = 0.88 + 0.12 * np.sin(_phase(5.2, t))
    return _normalize(_lowpass((bell + tine + body) * trem, 7200.0))


def ep_wurli(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    env = _adsr_env(t, 0.006, 0.12, 0.56, 0.11)
    reed = signal.sawtooth(_phase(f, t), width=0.58) * 0.55
    bite = np.sin(_phase(f * 2.0, t) + 1.9 * np.sin(_phase(f * 3.0, t))) * np.exp(-t / 0.22) * 0.32
    trem = 0.78 + 0.22 * np.sin(_phase(6.0, t))
    return _normalize(np.tanh(_lowpass((reed + bite) * env * trem, 5200.0) * 1.7))


def key_organ(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    drawbars = [(0.5, 0.25), (1.0, 0.85), (1.5, 0.18), (2.0, 0.42), (3.0, 0.20), (4.0, 0.10)]
    y = np.zeros_like(t)
    for ratio, amp in drawbars:
        y += amp * np.sin(_phase(f * ratio, t))
    vibrato = np.sin(_phase(5.7, t)) * 0.0035
    y += 0.18 * np.sin(_phase(f * 2.0, t + vibrato))
    return _normalize(_lowpass(y * _asr_env(t, 0.018, 0.030), 8200.0))


def celesta(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    partials = [(1.0, 0.70, 0.85), (2.0, 0.35, 0.58), (3.0, 0.18, 0.36), (5.0, 0.11, 0.22)]
    bell = _additive(t, f, partials)
    chime = np.sin(_phase(f * 2.98, t) + 0.8 * np.sin(_phase(f * 7.0, t))) * np.exp(-t / 0.34) * 0.24
    return _normalize(_highpass((bell + chime) * _exp_env(t, 1.1, 0.002), 120.0))


def marimba(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    partials = [(1.0, 0.90, 0.34), (3.92, 0.38, 0.16), (9.54, 0.16, 0.07)]
    bar = _additive(t, f, partials)
    knock = _bandpass(_noise(t.size, _seed(f, 31)), 500.0, 2400.0) * np.exp(-t / 0.008) * 0.18
    return _normalize(_lowpass((bar + knock) * _exp_env(t, 0.48, 0.002), 7600.0))


def kalimba(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    tine = _karplus(f, t.size, 0.989, 0.78, _seed(f, 41)) * np.exp(-t / 0.72)
    ping = np.sin(_phase(f * 2.04, t)) * np.exp(-t / 0.16) * 0.38
    buzz = signal.square(_phase(f * 6.0, t), duty=0.18) * np.exp(-t / 0.055) * 0.06
    return _normalize(_lowpass(tine + ping + buzz, 9200.0))


def vibraphone(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    partials = [(1.0, 0.85, 1.35), (2.01, 0.26, 0.92), (3.98, 0.14, 0.62), (6.02, 0.07, 0.42)]
    trem = 0.72 + 0.28 * np.sin(_phase(6.8, t))
    motor = np.sin(_phase(f, t + 0.0009 * np.sin(_phase(5.1, t)))) * np.exp(-t / 1.7) * 0.18
    return _normalize((_additive(t, f, partials) + motor) * _exp_env(t, 1.8, 0.005) * trem)


def pad_warm(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    detune = [0.995, 1.0, 1.006]
    y = np.zeros_like(t)
    for i, mul in enumerate(detune):
        y += signal.sawtooth(_phase(f * mul, t + i * 0.00013), width=0.54) * 0.22
        y += np.sin(_phase(f * 2.0 * mul, t)) * 0.10
    lfo = 0.86 + 0.14 * np.sin(_phase(0.38, t))
    env = _asr_env(t, 0.18, 0.35)
    return _normalize(_lowpass(y * env * lfo, min(4200.0, max(900.0, f * 8.0))))


def pad_glass(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    env = _asr_env(t, 0.24, 0.42)
    fm_env = 1.6 + 0.7 * np.sin(_phase(0.21, t))
    glass = np.sin(_phase(f, t) + fm_env * np.sin(_phase(f * 2.5, t))) * 0.55
    high = np.sin(_phase(f * 3.01, t) + 0.8 * np.sin(_phase(f * 4.0, t))) * 0.22
    shimmer = np.sin(_phase(f * 5.02, t + 0.0005 * np.sin(_phase(0.7, t)))) * 0.10
    return _normalize(_highpass((glass + high + shimmer) * env, 180.0))


def lead_square(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    width = 0.50 + 0.08 * np.sin(_phase(4.3, t))
    osc = signal.square(_phase(f, t), duty=np.clip(width, 0.08, 0.92))
    sub = signal.square(_phase(f * 0.5, t), duty=0.50) * 0.28
    env = _asr_env(t, 0.008, 0.030)
    filtered = _lowpass(osc + sub, min(9800.0, max(1200.0, f * 10.0)))
    return _normalize(np.tanh(filtered * env * 1.35))


def lead_saw(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    osc = signal.sawtooth(_phase(f * 0.997, t)) * 0.42
    osc += signal.sawtooth(_phase(f * 1.004, t + 0.0002)) * 0.42
    osc += np.sin(_phase(f * 2.0, t)) * 0.16
    env = _asr_env(t, 0.006, 0.040)
    sweep = min(12000.0, max(1600.0, f * (7.0 + 4.0 * np.exp(-min(float(dur), 1.0) / 0.45))))
    return _normalize(np.tanh(_lowpass(osc * env, sweep) * 1.5))


def lead_fm_sine(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    env = _asr_env(t, 0.004, 0.035)
    idx = 4.8 * np.exp(-t / 0.22) + 0.9
    mod = np.sin(_phase(f * 2.0, t)) * idx
    tone = np.sin(_phase(f, t) + mod)
    edge = np.sin(_phase(f * 3.0, t) + 1.5 * np.sin(_phase(f * 1.5, t))) * np.exp(-t / 0.18) * 0.25
    return _normalize(np.tanh((tone + edge) * env * 1.25))


def harp_pluck(freq: float, dur: float) -> np.ndarray:
    f = _freq(freq)
    t = _time(dur)
    string = _karplus(f, t.size, 0.995, 0.68, _seed(f, 51)) * np.exp(-t / 1.65)
    octave = _karplus(f * 2.0, t.size, 0.991, 0.54, _seed(f, 52)) * np.exp(-t / 0.95) * 0.20
    resonance = np.sin(_phase(f * 0.5, t)) * np.exp(-t / 1.1) * 0.12
    return _normalize(_lowpass(string + octave + resonance, 8800.0))

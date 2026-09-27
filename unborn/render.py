"""Mix note events into a stereo master and write audio. WAV via the stdlib
(no audioop dependency, which Python 3.13+ removed); mp3 via an ffmpeg
subprocess. velocity scales amplitude; each voice is synthesized on the fly,
placed by its track's `pan`, and the master is brought to a target loudness
(BS.1770) instead of a peak -- so every sculpture in the catalog sits at the
same perceived level."""
import subprocess
import wave

import numpy as np
from scipy import signal

from .drums import DRUM_VOICES
from .loudness import normalize
from .sequencer import NoteEvent
from .soundbank import SOUNDBANK
from .synth import SR, VOICES, midi_to_freq

ALL_VOICES = {**VOICES, **DRUM_VOICES, **SOUNDBANK}
# default for tracks that don't say `duck` themselves
DUCKABLE = {"bass", "subbass", "bell", "harmonic", "pad"}
MASTER = {"lufs": -14.0, "ceiling_db": -1.0}
VOICE_GAIN = {
    "kick": 0.85, "subbass": 0.5, "bass": 0.7, "hat": 0.55, "hat_open": 0.45,
    "clap": 0.7, "bell": 1.7, "harmonic": 1.5, "pad": 2.0,
}


def reverb_ir(decay: float = 1.8, seed: int = 1) -> np.ndarray:
    n = int(SR * decay)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (decay * SR / 5))
    return ir / (np.sqrt(np.sum(ir ** 2)) or 1.0)


def reverb(x: np.ndarray, amount: float = 0.22, decay: float = 1.8) -> np.ndarray:
    """Stereo: each channel gets its own decorrelated tail (seeds 1 and 2), which
    is what makes a reverb read as space rather than as a mono echo."""
    x = _stereo(x)
    wet = np.stack([np.asarray(signal.fftconvolve(x[:, ch], reverb_ir(decay, 1 + ch)),
                               dtype=np.float64)[: len(x)] for ch in range(2)], axis=1)
    return (1.0 - amount) * x + amount * wet


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.repeat(x[:, None], 2, axis=1) if x.ndim == 1 else x


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan law: centre is -3 dB per side, hard L/R is unity."""
    theta = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4
    return float(np.cos(theta)), float(np.sin(theta))


def ducks(e: NoteEvent) -> bool:
    return e.duck if e.duck is not None else e.voice in DUCKABLE


def resolve_voice(name: str, freq: float, dur: float) -> np.ndarray:
    if name.startswith("sample:"):
        from .sampler import render_sample
        return render_sample(name.split(":", 1)[1], freq, dur)
    fn = ALL_VOICES.get(name, VOICES["bell"])
    return fn(freq, dur)


def _bus(events: list[NoteEvent], n: int) -> np.ndarray:
    """Sum events into an (n, 2) buffer, each placed by its pan."""
    buf = np.zeros((n, 2), dtype=np.float64)
    for e in events:
        left, right = pan_gains(e.pan)
        gain = VOICE_GAIN.get(e.voice, 1.0)
        wave_data = resolve_voice(e.voice, midi_to_freq(e.note), e.duration)
        if e.fx:
            from .fx import apply_fx
            wave_data = apply_fx(wave_data, e.fx)
        wave_data = wave_data * (e.velocity / 127.0) * gain
        start = int(e.time * SR)
        stop = min(start + len(wave_data), n)
        seg = wave_data[: stop - start]
        buf[start:stop, 0] += seg * left
        buf[start:stop, 1] += seg * right
    return buf


def _duck_envelope(kick_times: list[float], n: int, amount: float, release: float) -> np.ndarray:
    env = np.ones(n, dtype=np.float64)
    rel = max(1, int(release * SR))
    recover = 1.0 - (1.0 - amount) * np.exp(-np.arange(rel) / (rel / 4))
    for kt in kick_times:
        i = int(kt * SR)
        seg = min(rel, n - i)
        if seg > 0:
            env[i:i + seg] = np.minimum(env[i:i + seg], recover[:seg])
    return env


def mix(events: list[NoteEvent], tail: float = 1.5, sidechain: dict | None = None,
        rev: dict | None = None, master: dict | None = None,
        receipts: dict | None = None) -> np.ndarray:
    """Stereo master, (n, 2). `receipts`, if given, is filled with the measured
    loudness before/after so the caller can print facts rather than intent."""
    if not events:
        return np.zeros((SR, 2), dtype=np.float64)
    end = max(e.time + e.duration for e in events) + tail
    n = int(SR * end) + 1
    if sidechain:
        src = sidechain.get("source_voice", "kick")
        kick_times = [e.time for e in events if e.voice == src]
        ducked = [e for e in events if ducks(e)]
        dry = [e for e in events if not ducks(e)]
        env = _duck_envelope(kick_times, n, sidechain.get("amount", 0.45),
                             sidechain.get("release", 0.18))
        out = _bus(dry, n) + _bus(ducked, n) * env[:, None]
    else:
        out = _bus(events, n)
    if rev:
        out = reverb(out, rev.get("amount", 0.22), rev.get("decay", 1.8))
    m = {**MASTER, **(master or {})}
    out, facts = normalize(out, m["lufs"], m["ceiling_db"])
    if receipts is not None:
        receipts.update(facts)
    return out


def write_wav(path: str, samples: np.ndarray) -> None:
    """1-D writes mono (the cues), (n, 2) writes stereo (the sculptures)."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "w") as w:
        w.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def to_mp3(wav_path: str) -> str | None:
    mp3_path = wav_path[:-4] + ".mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "4", mp3_path],
            check=True, capture_output=True,
        )
        return mp3_path
    except Exception as exc:
        print(f"  (mp3 skipped: {exc})")
        return None

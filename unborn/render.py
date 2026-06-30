"""Mix note events into a master buffer and write audio. WAV via the stdlib
(no audioop dependency, which Python 3.13+ removed); mp3 via an ffmpeg
subprocess. velocity scales amplitude; each voice is synthesized on the fly."""
import subprocess
import wave

import numpy as np
from scipy import signal

from .drums import DRUM_VOICES
from .sequencer import NoteEvent
from .soundbank import SOUNDBANK
from .synth import SR, VOICES, midi_to_freq

ALL_VOICES = {**VOICES, **DRUM_VOICES, **SOUNDBANK}
DUCKABLE = {"bass", "subbass", "bell", "harmonic", "pad"}
VOICE_GAIN = {
    "kick": 0.85, "subbass": 0.5, "bass": 0.7, "hat": 0.55, "hat_open": 0.45,
    "clap": 0.7, "bell": 1.7, "harmonic": 1.5, "pad": 2.0,
}


def reverb(x: np.ndarray, amount: float = 0.22, decay: float = 1.8) -> np.ndarray:
    n = int(SR * decay)
    rng = np.random.default_rng(1)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (decay * SR / 5))
    ir /= np.sqrt(np.sum(ir ** 2)) or 1.0
    wet = np.asarray(signal.fftconvolve(x, ir), dtype=np.float64)[: len(x)]
    return (1.0 - amount) * x + amount * wet


def resolve_voice(name: str, freq: float, dur: float) -> np.ndarray:
    if name.startswith("sample:"):
        from .sampler import render_sample
        return render_sample(name.split(":", 1)[1], freq, dur)
    fn = ALL_VOICES.get(name, VOICES["bell"])
    return fn(freq, dur)


def _bus(events: list[NoteEvent], n: int) -> np.ndarray:
    buf = np.zeros(n, dtype=np.float64)
    for e in events:
        gain = VOICE_GAIN.get(e.voice, 1.0)
        wave_data = resolve_voice(e.voice, midi_to_freq(e.note), e.duration)
        if e.fx:
            from .fx import apply_fx
            wave_data = apply_fx(wave_data, e.fx)
        wave_data = wave_data * (e.velocity / 127.0) * gain
        start = int(e.time * SR)
        stop = min(start + len(wave_data), n)
        buf[start:stop] += wave_data[: stop - start]
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
        rev: dict | None = None) -> np.ndarray:
    if not events:
        return np.zeros(SR, dtype=np.float64)
    end = max(e.time + e.duration for e in events) + tail
    n = int(SR * end) + 1
    if sidechain:
        src = sidechain.get("source_voice", "kick")
        kick_times = [e.time for e in events if e.voice == src]
        ducked = [e for e in events if e.voice in DUCKABLE]
        dry = [e for e in events if e.voice not in DUCKABLE]
        env = _duck_envelope(kick_times, n, sidechain.get("amount", 0.45),
                             sidechain.get("release", 0.18))
        master = _bus(dry, n) + _bus(ducked, n) * env
    else:
        master = _bus(events, n)
    if rev:
        master = reverb(master, rev.get("amount", 0.22), rev.get("decay", 1.8))
    peak = np.max(np.abs(master))
    if peak > 0:
        master = master / peak * 0.89
    return master


def write_wav(path: str, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "w") as w:
        w.setnchannels(1)
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

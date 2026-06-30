"""Mix note events into a master buffer and write audio. WAV via the stdlib
(no audioop dependency, which Python 3.13+ removed); mp3 via an ffmpeg
subprocess. velocity scales amplitude; each voice is synthesized on the fly."""
import subprocess
import wave

import numpy as np

from .sequencer import NoteEvent
from .synth import SR, midi_to_freq, render_voice


def mix(events: list[NoteEvent], tail: float = 1.5) -> np.ndarray:
    if not events:
        return np.zeros(SR, dtype=np.float64)
    end = max(e.time + e.duration for e in events) + tail
    master = np.zeros(int(SR * end) + 1, dtype=np.float64)
    for e in events:
        wave_data = render_voice(e.voice, midi_to_freq(e.note), e.duration) * (e.velocity / 127.0)
        start = int(e.time * SR)
        stop = start + len(wave_data)
        if stop > len(master):
            wave_data = wave_data[: len(master) - start]
            stop = len(master)
        master[start:stop] += wave_data
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

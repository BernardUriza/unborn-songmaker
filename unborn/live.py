"""Mode 2 -- the sculpture never stops.

Korda's kinetic sculptures have no end; `bars` was always the artificial cut we
made so an mp3 could exist. This module removes the cut: the sequencer walks a
sliding window of ticks forever, a producer thread renders each window into an
audio chunk, and PortAudio's callback does nothing but copy a ready buffer to the
card. Two things make that seamless -- an overlap-add `carry` so notes longer
than a chunk survive the boundary, and a stateful convolution reverb whose tail
crosses chunks instead of being truncated at each one.

The offline path (render.mix) normalises to the global peak, which is impossible
while streaming: per-chunk normalisation would pump. Here the gain is calibrated
once from a priming window and then held fixed, with a soft-knee limiter as the
only nonlinearity."""
import queue
import subprocess
import threading

import numpy as np
from scipy import signal

from .render import _bus, reverb_ir
from .synth import SR


class StreamingReverb:
    """render.reverb's impulse response, applied by overlap-add so the tail
    carries across chunk boundaries instead of being cut at every one."""

    def __init__(self, amount: float = 0.22, decay: float = 1.8):
        self.amount = amount
        self.ir = np.stack([reverb_ir(decay, 1), reverb_ir(decay, 2)], axis=1)  # L, R
        self.tail = np.zeros((len(self.ir) - 1, 2), dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        wet_full = np.stack([np.asarray(signal.fftconvolve(x[:, ch], self.ir[:, ch]),
                                        dtype=np.float64) for ch in range(2)], axis=1)
        wet_full[:len(self.tail)] += self.tail
        wet = wet_full[:len(x)]
        self.tail = wet_full[len(x):].copy()
        return (1.0 - self.amount) * x + self.amount * wet


class LiveRenderer:
    """Walks the sequencer forever, one tick-window at a time."""

    def __init__(self, seq, spec: dict, chunk_ticks: int = 96, tail_sec: float = 6.0):
        self.seq = seq
        self.spec = spec
        self.beats_per_bar = spec.get("beats_per_bar", 4)
        self.chunk_ticks = chunk_ticks
        self.tail_len = int(SR * tail_sec)
        self.spt = seq._seconds_per_tick()
        self.cycle_ticks = (spec.get("bars", 0) or 0) * self.beats_per_bar * seq.ticks_per_beat or None
        rev = spec.get("reverb") or {}
        self.reverb = StreamingReverb(rev.get("amount", 0.22), rev.get("decay", 1.8))
        self.cursor = 0
        self.carry = np.zeros((self.tail_len, 2), dtype=np.float64)
        self.gain = 1.0

    def _sample_at(self, tick: int) -> int:
        return int(round(tick * self.spt * SR))

    def _raw_chunk(self) -> np.ndarray:
        t0, t1 = self.cursor, self.cursor + self.chunk_ticks
        s0, s1 = self._sample_at(t0), self._sample_at(t1)
        n = s1 - s0
        events = self.seq.events_in(t0, t1, self.beats_per_bar, self.cycle_ticks)
        for e in events:
            e.time -= s0 / SR
        buf = _bus(events, n + self.tail_len)
        buf[:self.tail_len] += self.carry
        self.carry = buf[n:n + self.tail_len].copy()
        self.cursor = t1
        return buf[:n]

    def calibrate(self, windows: int = 24) -> float:
        peak = 0.0
        for _ in range(windows):
            peak = max(peak, float(np.max(np.abs(self.reverb.process(self._raw_chunk())))) )
        self.cursor = 0
        self.carry[:] = 0.0
        self.reverb.tail[:] = 0.0
        self.gain = 0.82 / peak if peak > 0 else 1.0
        return self.gain

    def chunk(self) -> np.ndarray:
        x = self.reverb.process(self._raw_chunk()) * self.gain
        k = 0.85
        over = np.abs(x) > k
        x[over] = np.sign(x[over]) * (k + (1 - k) * np.tanh((np.abs(x[over]) - k) / (1 - k)))
        return x.astype(np.float32)


def _producer(renderer: LiveRenderer, q: queue.Queue, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            q.put(renderer.chunk(), timeout=1.0)
        except queue.Full:
            continue


def play(renderer: LiveRenderer, buffer_chunks: int = 12, minutes: float | None = None) -> None:
    import sounddevice as sd
    q: queue.Queue = queue.Queue(maxsize=buffer_chunks)
    stop = threading.Event()
    for _ in range(buffer_chunks // 2):
        q.put(renderer.chunk())
    thread = threading.Thread(target=_producer, args=(renderer, q, stop), daemon=True)
    thread.start()
    pending = np.zeros((0, 2), dtype=np.float32)
    underruns = 0

    def callback(outdata, frames, time_info, status):
        nonlocal pending, underruns
        if status.output_underflow:
            underruns += 1
        while len(pending) < frames:
            try:
                pending = np.concatenate([pending, q.get_nowait()])
            except queue.Empty:
                underruns += 1
                outdata[:len(pending)] = pending
                outdata[len(pending):] = 0.0
                pending = np.zeros((0, 2), dtype=np.float32)
                return
        outdata[:] = pending[:frames]
        pending = pending[frames:]

    with sd.OutputStream(samplerate=SR, channels=2, dtype="float32",
                         blocksize=2048, callback=callback):
        try:
            stop.wait(timeout=minutes * 60 if minutes else None)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
    print(f"  underruns: {underruns}")


def pipe(renderer: LiveRenderer, args: list[str], minutes: float | None = None) -> None:
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR),
                             "-ac", "2", "-i", "pipe:0", *args], stdin=subprocess.PIPE)
    chunks = int(minutes * 60 / (renderer.chunk_ticks * renderer.spt)) if minutes else None
    written = 0
    assert proc.stdin is not None
    try:
        while chunks is None or written < chunks:
            proc.stdin.write(renderer.chunk().tobytes())
            written += 1
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        proc.stdin.close()
        proc.wait()

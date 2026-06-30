"""The kinetic sculpture. Steps every track at its own quantized clock; a track
fires a note when its current step is non-zero. Modulator tracks reshape their
targets each step -- position (the phase slip), transpose, velocity, mute -- so
the music emerges from the rule network, exactly as Korda's Polymeter does."""
from dataclasses import dataclass

from .track import (MOD_MUTE, MOD_NOTE, MOD_POSITION, MOD_VELOCITY, MODULATOR,
                    Modulation, Track)


@dataclass
class NoteEvent:
    time: float
    note: int
    velocity: int
    duration: float
    voice: str
    fx: dict | None = None


class Sequencer:
    def __init__(self, tracks: list[Track], modulations: list[Modulation],
                 tempo: float = 120.0, ticks_per_beat: int = 24):
        self.tracks = tracks
        self.modulations = modulations
        self.tempo = tempo
        self.ticks_per_beat = ticks_per_beat

    def _seconds_per_tick(self) -> float:
        return 60.0 / (self.tempo * self.ticks_per_beat)

    def _position_shift(self, target_index: int, global_step: int) -> int:
        shift = 0
        for mod in self.modulations:
            if mod.type == MOD_POSITION and mod.target == target_index:
                src = self.tracks[mod.source]
                shift += src.step_at(global_step) - src.note
        return shift

    def _mod_value(self, mod_type: str, target_index: int, global_step: int) -> int:
        value = 0
        for mod in self.modulations:
            if mod.type == mod_type and mod.target == target_index:
                src = self.tracks[mod.source]
                value += src.step_at(global_step)
        return value

    def _is_muted(self, target_index: int, global_step: int) -> bool:
        for mod in self.modulations:
            if mod.type == MOD_MUTE and mod.target == target_index:
                if self.tracks[mod.source].step_at(global_step) > 0:
                    return True
        return False

    def run(self, bars: int = 4, beats_per_bar: int = 4) -> list[NoteEvent]:
        spt = self._seconds_per_tick()
        events: list[NoteEvent] = []
        for ti, track in enumerate(self.tracks):
            if track.type == MODULATOR or track.mute:
                continue
            total_ticks = bars * beats_per_bar * self.ticks_per_beat
            bar_ticks = beats_per_bar * self.ticks_per_beat
            enter_tick = track.enter * bar_ticks
            exit_tick = track.exit * bar_ticks if track.exit is not None else total_ticks
            tick = track.offset
            global_step = 0
            while tick < total_ticks:
                if not (enter_tick <= tick < exit_tick):
                    tick += track.quant
                    global_step += 1
                    continue
                if self._is_muted(ti, global_step):
                    tick += track.quant
                    global_step += 1
                    continue
                read = global_step + self._position_shift(ti, global_step)
                vel = track.step_at(read)
                if vel > 0:
                    note = track.note + self._mod_value(MOD_NOTE, ti, global_step)
                    velocity = min(127, max(1, vel + track.velocity
                                            + self._mod_value(MOD_VELOCITY, ti, global_step)))
                    swing = track.swing if (global_step % 2 == 1) else 0
                    events.append(NoteEvent(
                        time=(tick + swing) * spt,
                        note=note,
                        velocity=velocity,
                        duration=track.quant * spt * 0.9,
                        voice=track.voice,
                        fx=track.fx,
                    ))
                tick += track.quant
                global_step += 1
        events.sort(key=lambda e: e.time)
        return events

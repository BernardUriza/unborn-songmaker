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
    pan: float = 0.0
    duck: bool | None = None


class Sequencer:
    def __init__(self, tracks: list[Track], modulations: list[Modulation],
                 tempo: float = 120.0, ticks_per_beat: int = 24):
        self.tracks = tracks
        self.modulations = modulations
        self.tempo = tempo
        self.ticks_per_beat = ticks_per_beat

    def _seconds_per_tick(self) -> float:
        return 60.0 / (self.tempo * self.ticks_per_beat)

    def _mods(self, mod_type: str, target_index: int):
        for mod in self.modulations:
            if mod.type == mod_type and mod.target == target_index:
                src = self.tracks[mod.source]
                if not src.mute:  # Korda: a muted modulator is simply ignored
                    yield src

    def _mod_value(self, mod_type: str, target_index: int, tick: int) -> int:
        """Sum of the modulators' step values at absolute `tick`, each read on
        its own clock (CSequencer::SumModulations). For MOD_POSITION the source's
        `note` is the resting centre, so a modulator at note 0 shifts by its raw
        step value."""
        value = 0
        for src in self._mods(mod_type, target_index):
            value += src.step_at(src.step_index(tick))
            if mod_type == MOD_POSITION:
                value -= src.note
        return value

    def _is_muted(self, target_index: int, tick: int) -> bool:
        return any(src.step_at(src.step_index(tick)) > 0
                   for src in self._mods(MOD_MUTE, target_index))

    def events_in(self, start_tick: int, end_tick: int, beats_per_bar: int = 4,
                  cycle_ticks: int | None = None) -> list[NoteEvent]:
        """Every note whose step falls in [start_tick, end_tick). The window is the
        only thing bounded -- `global_step` keeps counting from the absolute origin,
        so polymeter phase and modulations stay coherent across any number of
        windows. `cycle_ticks` folds each track's enter/exit window modulo a cycle:
        the arrangement recurs while the step phase keeps drifting, which is how a
        Korda sculpture runs forever without ever repeating itself."""
        spt = self._seconds_per_tick()
        bar_ticks = beats_per_bar * self.ticks_per_beat
        events: list[NoteEvent] = []
        for ti, track in enumerate(self.tracks):
            if track.type == MODULATOR or track.mute:
                continue
            enter_tick = track.enter * bar_ticks
            exit_tick = track.exit * bar_ticks if track.exit is not None else None
            first = max(0, -((track.offset - start_tick) // track.quant))
            global_step = first
            tick = track.offset + first * track.quant
            while tick < end_tick:
                pos = tick % cycle_ticks if cycle_ticks else tick
                live = pos >= enter_tick and (exit_tick is None or pos < exit_tick)
                if live and not self._is_muted(ti, tick):
                    read = global_step + self._mod_value(MOD_POSITION, ti, tick)
                    vel = track.step_at(read)
                    if vel > 0:
                        note = track.note + self._mod_value(MOD_NOTE, ti, tick)
                        velocity = min(127, max(1, vel + track.velocity
                                                + self._mod_value(MOD_VELOCITY, ti, tick)))
                        swing = track.swing if (global_step % 2 == 1) else 0
                        events.append(NoteEvent(
                            time=(tick + swing) * spt,
                            note=note,
                            velocity=velocity,
                            duration=track.quant * spt * 0.9,
                            voice=track.voice,
                            fx=track.fx,
                            pan=track.pan,
                            duck=track.duck,
                        ))
                tick += track.quant
                global_step += 1
        events.sort(key=lambda e: e.time)
        return events

    def run(self, bars: int = 4, beats_per_bar: int = 4) -> list[NoteEvent]:
        total_ticks = bars * beats_per_bar * self.ticks_per_beat
        return self.events_in(0, total_ticks, beats_per_bar)

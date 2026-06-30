"""Faithful port of Chris Korda's Polymeter track model (TrackDef.h,
victimofleisure/Polymeter, GPL-3.0). The CODE is a clean-room reimplementation;
only the algorithm -- which is not copyrightable -- is inherited.

A Track is a loop of `steps` with its own `length`. When tracks have relatively
prime lengths their phases slip against each other: that is polymeter. A
Modulation lets one track drive a property of another every step, which is how
Korda's 'kinetic sculptures' generate music instead of being written."""
from dataclasses import dataclass, field

NOTE = "note"
MODULATOR = "modulator"

MOD_MUTE = "mute"
MOD_NOTE = "note"
MOD_VELOCITY = "velocity"
MOD_POSITION = "position"


@dataclass
class Track:
    name: str = ""
    type: str = NOTE
    note: int = 64
    length: int = 16
    quant: int = 30
    offset: int = 0
    velocity: int = 0
    voice: str = "bell"
    steps: list[int] = field(default_factory=list)
    mute: bool = False

    def step_at(self, index: int) -> int:
        if not self.steps:
            return 0
        return self.steps[index % len(self.steps)]


@dataclass
class Modulation:
    type: str
    source: int
    target: int

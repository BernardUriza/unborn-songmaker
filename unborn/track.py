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
    length: int = 0  # 0 = derive from steps
    quant: int = 30
    offset: int = 0
    swing: int = 0
    velocity: int = 0
    voice: str = "bell"
    steps: list[int] = field(default_factory=list)
    mute: bool = False
    fx: dict | None = None
    enter: int = 0
    exit: int | None = None
    # render hints carried by the track (Korda's tracks carry a MIDI channel the
    # same way): stereo position -1..1, and whether the sidechain ducks this voice
    # (None = the render layer's default for that voice).
    pan: float = 0.0
    duck: bool | None = None

    def __post_init__(self) -> None:
        # Korda: the Length property IS the size of the step array (Track.h
        # GetLength). Here `length` wins: shorter truncates, longer pads with rests.
        if self.length < 1:
            self.length = len(self.steps) or 16
        if len(self.steps) != self.length:
            self.steps = (list(self.steps) + [0] * self.length)[: self.length]

    def step_at(self, index: int) -> int:
        if not self.steps:
            return 0
        return self.steps[index % self.length]

    def step_index(self, tick: int) -> int:
        """Which step this track is on at an absolute tick -- its OWN clock
        (offset, quant, length), regardless of who is asking. Port of
        CTrack::GetStepIndex; this is how a modulator is read by its target."""
        return ((tick - self.offset) // self.quant) % self.length


@dataclass
class Modulation:
    type: str
    source: int
    target: int

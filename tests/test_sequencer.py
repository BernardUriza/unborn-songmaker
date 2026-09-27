"""Contract-layer invariants, checked against Korda's Polymeter semantics."""
from unborn.spec import euclid
from unborn.track import Modulation, Track
from unborn.sequencer import Sequencer


def test_euclid_spreads_pulses():
    assert euclid(3, 8, 1) == [0, 0, 1, 0, 0, 1, 0, 1]
    assert sum(euclid(5, 13, 1)) == 5
    assert euclid(0, 4) == [0, 0, 0, 0]


def test_relatively_prime_lengths_realign_at_lcm():
    a = Track(steps=[1, 0, 0], quant=1)
    b = Track(steps=[1, 0, 0, 0, 0], quant=1)
    seq = Sequencer([a, b], [], ticks_per_beat=1)
    ticks = [round(e.time / seq._seconds_per_tick()) for e in seq.events_in(0, 30)]
    assert sorted(t for t in set(ticks) if ticks.count(t) == 2) == [0, 15]


def test_modulator_is_read_on_its_own_clock():
    # Korda CTrack::GetStepIndex: the source's quant decides when its step
    # changes, not the target's step counter. Source holds each value 4 ticks.
    src = Track(name="m", type="modulator", note=0, steps=[0, 5], quant=4)
    tgt = Track(name="t", note=64, steps=[100], quant=1)
    seq = Sequencer([tgt, src], [Modulation("note", 1, 0)], ticks_per_beat=1)
    notes = [e.note for e in seq.events_in(0, 12)]
    assert notes == [64] * 4 + [69] * 4 + [64] * 4


def test_length_is_the_loop():
    assert Track(steps=[1, 2, 3], length=5).steps == [1, 2, 3, 0, 0]
    assert Track(steps=[1, 2, 3], length=2).steps == [1, 2]
    assert Track(steps=[1, 2, 3]).length == 3

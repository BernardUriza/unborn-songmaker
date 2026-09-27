"""Contract-layer invariants, checked against Korda's Polymeter semantics."""
from unborn.spec import euclid
from unborn.track import Track
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


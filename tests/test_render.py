"""Render-layer receipts: the loudness meter against the standard's own
reference point, the limiter against its ceiling, the pan law's power."""
import numpy as np

from unborn.loudness import SR, integrated_lufs, limiter, normalize, true_peak_db
from unborn.render import pan_gains
from unborn.sequencer import NoteEvent
from unborn.render import _bus


def _sine(freq=997.0, seconds=3.0, amp=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def test_meter_matches_bs1770_reference():
    # BS.1770-4: a 0 dBFS 997 Hz sine in one channel reads -3.01 LUFS.
    stereo = np.stack([_sine(), np.zeros(int(SR * 3.0))], axis=1)
    assert abs(integrated_lufs(stereo) - (-3.01)) < 0.02
    assert abs(true_peak_db(stereo)) < 0.05
    assert integrated_lufs(np.zeros((SR, 2))) == float("-inf")


def test_limiter_holds_ceiling():
    hot = np.stack([_sine(amp=2.0)] * 2, axis=1)
    out = limiter(hot, ceiling_db=-1.0)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-9
    assert true_peak_db(out) <= -0.9


def test_normalize_reaches_target():
    quiet = np.stack([_sine(amp=0.05)] * 2, axis=1)
    out, facts = normalize(quiet, target_lufs=-14.0)
    assert abs(facts["lufs"] - (-14.0)) < 0.1
    assert facts["true_peak_db"] <= -0.9


def test_pan_law_is_constant_power():
    for pan in (-1.0, -0.5, 0.0, 0.3, 1.0):
        left, right = pan_gains(pan)
        assert abs(left ** 2 + right ** 2 - 1.0) < 1e-9
    assert pan_gains(-1.0) == (1.0, 0.0) or abs(pan_gains(-1.0)[1]) < 1e-12
    assert abs(pan_gains(1.0)[0]) < 1e-12


def test_bus_places_event_by_pan():
    e = NoteEvent(time=0.0, note=60, velocity=127, duration=0.1, voice="subbass", pan=1.0)
    buf = _bus([e], SR // 2)
    assert np.max(np.abs(buf[:, 0])) < 1e-9 and np.max(np.abs(buf[:, 1])) > 0.01

"""The catalog as a regression net. Every spec must validate, and the sequencer's
output for each spec is pinned by a digest of its note events -- so a change to the
contract layer shows exactly which sculptures it re-voices and which it leaves
alone. After an INTENDED change: `UNBORN_UPDATE_SNAPSHOTS=1 pytest`, listen to
the specs that moved, commit the new snapshot with the reason."""
import glob
import hashlib
import json
import os

import numpy as np
import pytest

from unborn.render import mix
from unborn.spec import sequencer_from, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = sorted(glob.glob(os.path.join(ROOT, "specs", "*.json")))
SNAPSHOT = os.path.join(ROOT, "tests", "snapshots", "events.json")
UPDATE = os.environ.get("UNBORN_UPDATE_SNAPSHOTS") == "1"


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _events(spec: dict):
    seq = sequencer_from(spec)
    return seq.run(bars=spec.get("bars", 4), beats_per_bar=spec.get("beats_per_bar", 4))


def _digest(events) -> dict:
    rows = [(round(e.time, 6), e.note, e.velocity, round(e.duration, 6), e.voice)
            for e in events]
    return {"events": len(rows),
            "sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]}


@pytest.fixture(scope="session")
def snapshot():
    data = _load(SNAPSHOT) if os.path.exists(SNAPSHOT) else {}
    yield data
    if UPDATE:
        with open(SNAPSHOT, "w") as f:
            json.dump(dict(sorted(data.items())), f, indent=2)
            f.write("\n")


@pytest.mark.parametrize("path", SPECS, ids=_stem)
def test_spec_validates(path):
    assert validate(_load(path), check_samples=False) == []


@pytest.mark.parametrize("path", SPECS, ids=_stem)
def test_events_match_snapshot(path, snapshot):
    got = _digest(_events(_load(path)))
    if UPDATE:
        snapshot[_stem(path)] = got
        return
    assert _stem(path) in snapshot, "no snapshot yet -- run with UNBORN_UPDATE_SNAPSHOTS=1"
    assert got == snapshot[_stem(path)]


@pytest.mark.parametrize("path", SPECS, ids=_stem)
def test_render_is_alive(path):
    """The ALIVE half of the audio verification discipline: the first 16 bars
    render finite, non-silent and not flat. Whether it sounds GOOD is the ear's job."""
    spec = _load(path)
    if any("sample not found" in e for e in validate(spec)):
        pytest.skip("local voice samples (gitignored) not present")
    seq = sequencer_from(spec)
    bpb = spec.get("beats_per_bar", 4)
    window = min(spec.get("bars", 4), 16) * bpb * seq.ticks_per_beat
    audio = mix(seq.events_in(0, window, bpb), sidechain=spec.get("sidechain"),
                rev=spec.get("reverb"))
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert np.all(np.isfinite(audio))
    mono = audio.mean(axis=1)
    assert np.sqrt(np.mean(mono ** 2)) > 1e-3
    frames = mono[: len(mono) // 4410 * 4410].reshape(-1, 4410)
    assert np.std(np.sqrt(np.mean(frames ** 2, axis=1))) > 1e-4
    assert np.max(np.abs(audio)) <= 10 ** (-1.0 / 20) + 1e-6  # limiter ceiling held

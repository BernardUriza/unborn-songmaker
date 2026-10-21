"""The UI's spec edits are textual splices: a track added is one new line, the
rest of the file is byte-identical, and add-then-remove restores the file
exactly. Every edit passes through validate() -- the UI can't save a bad spec."""
import glob
import json
import os

import pytest

from unborn.spec import SpecError
from unborn.specedit import _tracks_array, add_track, inline, remove_track

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = sorted(glob.glob(os.path.join(ROOT, "specs", "*.json")))
TRACK = {"voice": "marimba", "note": 67, "quant": 12, "length": 7,
         "euclid": {"pulses": 3, "length": 7, "velocity": 80}}


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize("path", SPECS, ids=[os.path.basename(p)[:-5] for p in SPECS])
def test_scanner_finds_every_track(path):
    text = _read(path)
    assert len(_tracks_array(text)[2]) == len(json.loads(text)["tracks"])


@pytest.mark.parametrize("path", SPECS, ids=[os.path.basename(p)[:-5] for p in SPECS])
def test_add_is_one_line_and_remove_restores(path):
    src = _read(path)
    new, saved = add_track(src, TRACK)
    assert json.loads(new)["tracks"][:-1] == json.loads(src)["tracks"]
    assert json.loads(new)["tracks"][-1] == saved
    extra = [l for l in new.splitlines() if l not in src.splitlines()]
    assert len(extra) <= 2  # the new line, plus the previous line gaining a comma
    assert remove_track(new, saved["name"]) == src


def test_name_defaults_to_voice_and_dedupes():
    src = _read(SPECS[0])
    new, saved = add_track(src, {**TRACK, "name": "", "exit": None, "fx": {}})
    assert saved["name"] == "marimba" and "exit" not in saved and "fx" not in saved
    assert list(saved) == ["name", "voice", "note", "quant", "length", "euclid"]
    _, again = add_track(new, TRACK)
    assert again["name"] == "marimba-2"
    with pytest.raises(SpecError, match="already exists"):
        add_track(new, {**TRACK, "name": "marimba"})


def test_bad_track_is_refused_with_every_error():
    src = _read(SPECS[0])
    with pytest.raises(SpecError) as e:
        add_track(src, {**TRACK, "voice": "nope", "note": 200})
    assert any("unknown voice" in x for x in e.value.errors)
    assert any("note must be" in x for x in e.value.errors)


def test_remove_refuses_to_orphan_a_modulation():
    src = _read(os.path.join(ROOT, "specs", "hvob.json"))
    with pytest.raises(SpecError, match="modulation"):
        remove_track(src, "drift")
    with pytest.raises(SpecError, match="no track named"):
        remove_track(src, "ghost")
    out = json.loads(remove_track(src, "rim"))
    assert "rim" not in [t["name"] for t in out["tracks"]]


def test_inline_is_house_style():
    assert inline({"name": "x", "steps": [1, 0], "euclid": {"pulses": 3}}) == \
        '{ "name": "x", "steps": [1, 0], "euclid": { "pulses": 3 } }'


def test_preview_and_voices(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from unborn import ui
    monkeypatch.setattr(ui, "PREVIEW", str(tmp_path))
    groups = {v["group"] for v in ui.voices()}
    assert {"synth", "drums", "tonal", "nature", "city", "drums_house"} <= groups
    with pytest.raises(LookupError):
        ui.preview("nope_voice", 60)
    wav = ui.preview("marimba", 67)
    data, sr = sf.read(wav)
    assert sr == 44100 and np.abs(data).max() > 0.05 and not np.isnan(data).any()
    assert ui.preview("marimba", 67) == wav  # cached

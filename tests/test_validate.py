"""validate() must name every silent mis-render instead of letting it sound."""
import copy
import json
import os

import pytest

from unborn.spec import SpecError, load, sequencer_from, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = json.load(open(os.path.join(ROOT, "specs", "techno.json")))


def _spec():
    return copy.deepcopy(BASE)


def test_collects_every_error_at_once():
    s = _spec()
    s["tracks"][0]["voice"] = "kik"
    s["tracks"][1]["exti"] = 3
    s["modulations"].append({"type": "pos", "source": 99, "target": 0})
    errors = validate(s)
    assert any("unknown voice 'kik'" in e for e in errors)
    assert any("unknown key 'exti'" in e for e in errors)
    assert any("type must be one of" in e for e in errors)
    assert any("source 99" in e for e in errors)


def test_windows_and_ranges():
    s = _spec()
    s["tracks"][0]["enter"], s["tracks"][0]["exit"] = 4, 2
    s["tracks"][1]["steps"] = [0, 200]
    s["tracks"][2]["swing"] = 99
    errors = validate(s)
    assert any("exit 2 must be > enter 4" in e for e in errors)
    assert any("0..127" in e for e in errors)
    assert any("swing" in e for e in errors)


def test_modulator_steps_may_be_signed():
    s = _spec()
    drift = next(t for t in s["tracks"] if t.get("type") == "modulator")
    drift["steps"] = [0, -7, 5]
    assert validate(s) == []


def test_load_raises_spec_error(tmp_path):
    s = _spec()
    s["tracks"][0]["voice"] = "kik"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(s))
    with pytest.raises(SpecError, match="kik"):
        load(str(p))


def test_modulations_by_name():
    s = _spec()
    by_index = sequencer_from(s).modulations
    names = [t["name"] for t in s["tracks"]]
    for m in s["modulations"]:
        m["source"], m["target"] = names[m["source"]] if isinstance(m["source"], int) else m["source"], \
            names[m["target"]] if isinstance(m["target"], int) else m["target"]
    assert validate(s) == []
    assert sequencer_from(s).modulations == by_index


def test_bad_name_refs():
    s = _spec()
    s["modulations"] = [{"type": "note", "source": "nope", "target": "kick"},
                        {"type": "note", "source": "kick", "target": "clap"}]
    errors = validate(s)
    assert any("'nope' is not a track name" in e for e in errors)
    assert any("is not type 'modulator'" in e for e in errors)
    s["tracks"][1]["name"] = "kick"
    s["modulations"] = [{"type": "position", "source": "drift", "target": "kick"}]
    assert any("names 2 tracks" in e for e in validate(s))


def test_pan_and_master_keys():
    s = _spec()
    s["tracks"][0]["pan"] = 2
    s["master"] = {"lufs": -16, "loud": 1}
    errors = validate(s)
    assert any("pan must be in -1..1" in e for e in errors)
    assert any("master: unknown key 'loud'" in e for e in errors)

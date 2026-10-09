"""Home foley path: videopipe's sfx-<session>.zip -> samples/<id>/take-NN.wav ->
`sample:<id>` plays the takes round-robin. Everything runs on a synthetic bundle
in tmp_path; the real samples/ folder (gitignored) is never touched."""
import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from unborn import sampler, spec as spec_mod
from unborn.render import _bus
from unborn.sequencer import NoteEvent
from unborn.sfx_import import SfxImportError, import_bundle, trim_take
from unborn.synth import SR

REC_SR = 48000


def _take(burst_at=0.5, burst_len=0.08, freq=1000.0, seed=0):
    """0.5 s room tone, a decaying burst, 0.5 s room tone -- what the recorder asks for."""
    rng = np.random.default_rng(seed)
    n = int(REC_SR * (burst_at + burst_len + 0.5))
    x = rng.normal(0, 10 ** (-60 / 20), n)
    t = np.arange(int(REC_SR * burst_len)) / REC_SR
    i = int(REC_SR * burst_at)
    x[i:i + len(t)] += 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 40)
    return x


def _wav_bytes(x):
    buf = io.BytesIO()
    sf.write(buf, x, REC_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _bundle(path, tasks, kind="videopipe-sfx"):
    manifest = {"kind": kind, "version": 1, "session": "receipts", "tasks": []}
    with zipfile.ZipFile(path, "w") as zf:
        for tid, takes in tasks.items():
            entries = []
            for take, status, data in takes:
                f = f"{tid}/take-{take:02d}.wav"
                zf.writestr(f, _wav_bytes(data))
                entries.append({"take": take, "file": f, "status": status,
                                "peakDbfs": -6.0, "clipped": False})
            manifest["tasks"].append({"id": tid, "takes": entries})
        zf.writestr("manifest.json", json.dumps(manifest))
    return path


def test_trim_starts_on_the_transient():
    x = _take(burst_at=0.5)
    cut, onset = trim_take(x, REC_SR)
    assert 0.48 <= onset <= 0.5  # a few ms of pre-roll, not the 0.5 s of room
    assert len(cut) < len(x) * 0.5
    assert np.argmax(np.abs(cut)) / REC_SR < 0.03
    assert cut[0] == 0.0 and abs(cut[-1]) < 1e-9  # faded both ends: no click


def test_import_writes_only_kept_takes_and_mirrors_the_bundle(tmp_path):
    samples = tmp_path / "samples"
    (samples / "stamp_hit").mkdir(parents=True)
    (samples / "stamp_hit" / "take-09.wav").write_bytes(_wav_bytes(_take()))  # stale import
    z = _bundle(tmp_path / "sfx-receipts.zip", {
        "stamp_hit": [(1, "keep", _take(seed=1)), (2, "pending", _take(seed=2)),
                      (4, "keep", _take(seed=4))],
        "pen_click": [(1, "pending", _take(seed=5))],
    })
    report = {r["task"]: r for r in import_bundle(str(z), samples_dir=str(samples))}
    assert report["stamp_hit"]["written"] == ["take-01.wav", "take-04.wav"]
    assert sorted(p.name for p in (samples / "stamp_hit").glob("*.wav")) == ["take-01.wav", "take-04.wav"]
    assert report["pen_click"]["written"] == [] and not (samples / "pen_click").exists()
    prov = json.loads((samples / "stamp_hit" / "takes.json").read_text())
    assert [t["take"] for t in prov["takes"]] == [1, 4]
    data, sr = sf.read(samples / "stamp_hit" / "take-01.wav")
    assert sr == REC_SR and len(data) < REC_SR * 0.5

    report = {r["task"]: r for r in import_bundle(str(z), samples_dir=str(samples),
                                                  include_pending=True)}
    assert report["pen_click"]["written"] == ["take-01.wav"]
    assert len(report["stamp_hit"]["written"]) == 3


def test_import_refuses_foreign_or_unsafe_bundles(tmp_path):
    with pytest.raises(SfxImportError, match="kind"):
        import_bundle(str(_bundle(tmp_path / "a.zip", {}, kind="voiceover")), str(tmp_path))
    with pytest.raises(SfxImportError, match="snake_case"):
        import_bundle(str(_bundle(tmp_path / "b.zip", {"../evil": [(1, "keep", _take())]})),
                      str(tmp_path))


def test_folder_of_takes_plays_round_robin_and_validates(tmp_path, monkeypatch):
    d = tmp_path / "paper_flip"
    d.mkdir()
    # two distinguishable takes: one positive DC block, one negative
    sf.write(d / "take-01.wav", np.full(2000, 0.5), SR, subtype="PCM_16")
    sf.write(d / "take-02.wav", np.full(2000, -0.5), SR, subtype="PCM_16")
    monkeypatch.setattr(sampler, "SAMPLES_DIR", str(tmp_path))
    monkeypatch.setattr(spec_mod, "SAMPLES_DIR", str(tmp_path))
    monkeypatch.setattr(sampler, "_cache", {})

    assert len(sampler.sample_takes("paper_flip")) == 2
    events = [NoteEvent(time=i * 0.1, note=60, velocity=127, duration=0.05,
                        voice="sample:paper_flip") for i in range(4)]
    buf = _bus(events, int(SR * 0.5))
    signs = [np.sign(buf[int(i * 0.1 * SR) + 10, 0]) for i in range(4)]
    assert signs == [1, -1, 1, -1]

    s = {"name": "t", "tempo": 72, "ticks_per_beat": 24, "bars": 1,
         "tracks": [{"name": "flip", "voice": "sample:paper_flip", "note": 60, "steps": [100]}]}
    assert spec_mod.validate(s) == []
    s["tracks"][0]["voice"] = "sample:nope"
    assert any("sample not found" in e for e in spec_mod.validate(s))

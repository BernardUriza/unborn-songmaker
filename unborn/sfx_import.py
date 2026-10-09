"""Home foley in: `sfx-<session>.zip` from videopipe's recorder (/sfx/record/<id>)
-> samples/<task-id>/take-NN.wav, usable in any spec as `"voice": "sample:<task-id>"`
(round-robin over the takes, see sampler.py).

The zip is the file boundary between the two repos -- nothing is imported from
videopipe. Layout it carries (manifest kind "videopipe-sfx", version 1):

    manifest.json                 tasks[].id, tasks[].takes[] {take, file, status,
                                  durationSeconds, peakDbfs, clipped}
    <task-id>/take-NN.wav         48 kHz mono 16-bit, ~0.5 s of room before/after

What the import does to each take, and why:
  - only status "keep" (Bernard's judgement in the recorder) unless asked;
  - trims the leading room tone to a few ms before the onset -- a sample on the
    beat grid must START on its transient, or every hit lands ~0.5 s late;
  - trims the trailing room tone after the decay, with a short fade;
  - the folder MIRRORS the bundle: old take-*.wav in it are replaced, so a take
    discarded since the last import does not linger in the rotation.
The originals stay in the zip; this is the playable cut."""
import io
import json
import os
import re
import zipfile

import numpy as np
import soundfile as sf

from .sampler import SAMPLES_DIR

MANIFEST_KIND = "videopipe-sfx"
TASK_ID = re.compile(r"^[a-z][a-z0-9_]*$")
FRAME_SEC = 0.005
PRE_ROLL_SEC = 0.010
TAIL_SEC = 0.060
FADE_IN_SEC = 0.002
FADE_OUT_SEC = 0.030


class SfxImportError(ValueError):
    pass


def trim_take(data: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Cut leading/trailing room tone. Returns (trimmed, onset_seconds_in_original).

    Threshold = the louder of 40 dB under the peak and 4x the room's level
    (median envelope of the first 50 ms, which the recorder asks to be silence),
    measured on 5 ms max-abs frames so a single noise spike can't fake an onset."""
    if data.size == 0:
        return data, 0.0
    frame = max(1, int(sr * FRAME_SEC))
    n_frames = int(np.ceil(len(data) / frame))
    padded = np.zeros(n_frames * frame)
    padded[: len(data)] = np.abs(data)
    env = padded.reshape(n_frames, frame).max(axis=1)
    peak = float(env.max())
    if peak <= 0:
        return data, 0.0
    room = float(np.median(env[: max(1, int(0.05 / FRAME_SEC))]))
    thresh = max(peak * 10 ** (-40 / 20), room * 4)
    loud = np.flatnonzero(env >= thresh)
    first, last = int(loud[0]), int(loud[-1])
    start = max(0, first * frame - int(sr * PRE_ROLL_SEC))
    stop = min(len(data), (last + 1) * frame + int(sr * TAIL_SEC))
    out = data[start:stop].astype(np.float64).copy()
    fi, fo = min(len(out), int(sr * FADE_IN_SEC)), min(len(out), int(sr * FADE_OUT_SEC))
    if fi:
        out[:fi] *= np.linspace(0.0, 1.0, fi)
    if fo:
        out[-fo:] *= np.linspace(1.0, 0.0, fo)
    return out, start / sr


def read_bundle(zip_path: str) -> tuple[zipfile.ZipFile, dict]:
    zf = zipfile.ZipFile(zip_path)
    try:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except KeyError as exc:
            raise SfxImportError(f"{zip_path}: no manifest.json -- not an sfx bundle") from exc
        if manifest.get("kind") != MANIFEST_KIND:
            raise SfxImportError(f"{zip_path}: manifest kind is {manifest.get('kind')!r}, "
                                 f"expected {MANIFEST_KIND!r}")
    except Exception:
        zf.close()
        raise
    return zf, manifest


def import_bundle(zip_path: str, samples_dir: str | None = None,
                  include_pending: bool = False) -> list[dict]:
    """Write the chosen takes of every task; returns one report dict per task."""
    root = samples_dir or SAMPLES_DIR
    zf, manifest = read_bundle(zip_path)
    wanted = {"keep", "pending"} if include_pending else {"keep"}
    report = []
    with zf:
        for task in manifest.get("tasks", []):
            tid = task.get("id", "")
            if not TASK_ID.match(tid):
                raise SfxImportError(f"task id {tid!r} is not snake_case -- refusing to write it")
            chosen = [t for t in task.get("takes", []) if t.get("status") in wanted]
            entry = {"task": tid, "written": [], "skipped": len(task.get("takes", [])) - len(chosen),
                     "clipped": [], "shadowed": os.path.isfile(os.path.join(root, f"{tid}.wav"))}
            if not chosen:
                report.append(entry)
                continue
            dest_dir = os.path.join(root, tid)
            os.makedirs(dest_dir, exist_ok=True)
            for old in os.listdir(dest_dir):
                if old.startswith("take-") and old.endswith(".wav"):
                    os.remove(os.path.join(dest_dir, old))
            provenance = []
            for t in chosen:
                take = int(t["take"])
                name = f"take-{take:02d}.wav"
                expected = f"{tid}/{name}"
                if t.get("file") != expected:
                    raise SfxImportError(f"{tid}: take {take} points at {t.get('file')!r}, "
                                         f"expected {expected!r}")
                data, sr = sf.read(io.BytesIO(zf.read(expected)), dtype="float64")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                cut, onset = trim_take(data, sr)
                sf.write(os.path.join(dest_dir, name), cut, sr, subtype="PCM_16")
                entry["written"].append(name)
                if t.get("clipped"):
                    entry["clipped"].append(name)
                provenance.append({"file": name, "take": take, "status": t.get("status"),
                                   "session": manifest.get("session"),
                                   "peakDbfs": t.get("peakDbfs"), "clipped": bool(t.get("clipped")),
                                   "trimmedFromSeconds": round(onset, 4),
                                   "durationSeconds": round(len(cut) / sr, 4)})
            with open(os.path.join(dest_dir, "takes.json"), "w") as fh:
                json.dump({"source": os.path.basename(zip_path), "takes": provenance}, fh, indent=2)
            report.append(entry)
    return report

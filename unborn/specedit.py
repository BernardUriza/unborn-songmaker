"""Editing a spec from the UI without touching the file's shape. The JSON on
disk is hand-written (one track per line, blank lines grouping sections) and
is the source of truth, so an edit is a textual splice inside the `tracks`
array: add = one new line before the closing bracket, remove = that track's
span cut out. Everything else stays byte-for-byte, which keeps the git diff
honest (one track, not a reformatted file). Every write is validated first --
the UI can never save a spec the engine would render wrong."""
import json

from .spec import SpecError, validate


def _tracks_array(text: str) -> tuple[int, int, list[tuple[int, int]]]:
    """(open_bracket, close_bracket, [(start, end) of each element])."""
    key = text.find('"tracks"')
    if key < 0:
        raise SpecError(["no 'tracks' array in file"])
    i = text.index("[", key)
    depth, in_str, esc = 0, False, False
    spans: list[tuple[int, int]] = []
    start = None
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
            if depth == 2:
                start = j
        elif c in "]}":
            depth -= 1
            if depth == 1 and start is not None:
                spans.append((start, j + 1))
                start = None
            elif depth == 0:
                return i, j, spans
    raise SpecError(["unterminated 'tracks' array"])


def inline(v) -> str:
    """One-line JSON in the catalog's house style: `{ "k": v, "k": [1, 2] }`."""
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(k)}: {inline(x)}" for k, x in v.items()) + " }"
    if isinstance(v, list):
        return "[" + ", ".join(inline(x) for x in v) + "]"
    return json.dumps(v)


# house order for a track line: identity, clock, window, mix, pattern, colour
KEY_ORDER = ["name", "type", "voice", "note", "quant", "length", "offset", "swing",
             "enter", "exit", "pan", "duck", "velocity", "steps", "euclid", "fx", "mute"]


def _clean(track: dict) -> dict:
    """Drop the empty values a form sends so the file only carries intent."""
    return {k: v for k, v in track.items() if v not in (None, "", {}, [])}


def _ordered(track: dict) -> dict:
    return {k: track[k] for k in KEY_ORDER if k in track} | {k: v for k, v in track.items() if k not in KEY_ORDER}


def unique_name(name: str, spec: dict) -> str:
    names = {t.get("name") for t in spec.get("tracks", [])}
    if name not in names:
        return name
    i = 2
    while f"{name}-{i}" in names:
        i += 1
    return f"{name}-{i}"


def _check(text: str) -> dict:
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise SpecError([f"edit produced invalid JSON: {e}"])
    errors = validate(spec)
    if errors:
        raise SpecError(errors)
    return spec


def add_track(text: str, track: dict) -> tuple[str, dict]:
    """Returns (new file text, the track as saved). Raises SpecError carrying
    every problem so one edit pass fixes them all."""
    spec = json.loads(text)
    track = _clean(track)
    if not track.get("name"):
        track["name"] = unique_name(str(track.get("voice", "track")).replace("sample:", ""), spec)
    elif track["name"] in {t.get("name") for t in spec.get("tracks", [])}:
        raise SpecError([f"a track named '{track['name']}' already exists"])
    track = _ordered(track)
    _, close, spans = _tracks_array(text)
    indent = "    "
    if spans:
        line_start = text.rfind("\n", 0, spans[-1][0]) + 1
        indent = text[line_start:spans[-1][0]]
        if indent.strip():  # element didn't start its own line; fall back
            indent = "    "
    before = text[:close].rstrip()
    sep = ("," if spans else "") + "\n" + indent
    pre = text[text.rfind("\n", 0, close) + 1:close]  # the bracket's own indent
    bracket = (pre if not pre.strip() else "") + text[close:]
    new_text = before + sep + inline(track) + "\n" + bracket
    return new_text, _check(new_text)["tracks"][-1]


def remove_track(text: str, name: str) -> str:
    spec = json.loads(text)
    idx = [t.get("name") for t in spec.get("tracks", [])]
    if name not in idx:
        raise SpecError([f"no track named '{name}'"])
    k = idx.index(name)
    open_, close, spans = _tracks_array(text)
    s, e = spans[k]
    # cut from the previous element's end (its comma + whitespace) to ours, or,
    # for the first element, from our start through the comma that follows us
    if k > 0:
        cut_s = spans[k - 1][1]
        cut_e = e
    else:
        cut_s = s
        cut_e = spans[k + 1][0] if len(spans) > 1 else e
    new_text = text[:cut_s] + text[cut_e:]
    _check(new_text)  # a modulation still naming it -> refuse, don't orphan
    return new_text

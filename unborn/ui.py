"""A local window onto the catalog -- the glass box made visible.

Lists every spec in `specs/`, plays its rendered mp3 from `out/`, re-renders on
demand, and draws the arrangement: each track as a bar over its `enter`/`exit`
window, with a playhead tied to the audio clock. Stdlib only (http.server); a
render runs `cli.py sculpture` in a subprocess so the server never blocks and a
crash in the engine can't take the page down.

Adding a sound: the page lists every registered voice (synth, drums, soundbank
modules, recorded samples), auditions one through the real render layer
(`_bus`: VOICE_GAIN, fx) as a short loudness-matched wav, and appends it to the
spec as an ordinary Track -- a textual splice that leaves the file's shape alone
(`specedit`). The spec stays the source of truth; the UI never bypasses
`validate()`, so a bad add comes back as the list of errors, not as a sound."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "specs")
OUT = os.path.join(ROOT, "out")
BRANDING = os.path.join(ROOT, "branding")
SAMPLES = os.path.join(ROOT, "samples")
PREVIEW = os.path.join(OUT, "preview")
PREVIEW_LUFS = -18.0  # every audition at the same level: what you compare is timbre

_jobs: dict[str, dict] = {}  # spec stem -> {"state": running|done|error, "log": str}
_lock = threading.Lock()


def _spec_path(stem: str) -> str | None:
    p = os.path.join(SPECS, f"{stem}.json")
    return p if os.path.dirname(os.path.abspath(p)) == SPECS and os.path.exists(p) else None


def catalog() -> list[dict]:
    items = []
    for f in sorted(os.listdir(SPECS)):
        if not f.endswith(".json"):
            continue
        stem = f[:-5]
        path = os.path.join(SPECS, f)
        try:
            spec = json.load(open(path))
        except (OSError, json.JSONDecodeError) as e:
            items.append({"stem": stem, "error": str(e)})
            continue
        name = spec.get("name") or stem
        mp3 = os.path.join(OUT, f"{name}.mp3")
        has = os.path.exists(mp3)
        bars = spec.get("bars", 4)
        items.append({
            "stem": stem,
            "name": name,
            "tempo": spec.get("tempo"),
            "bars": bars,
            "beats_per_bar": spec.get("beats_per_bar", 4),
            "mp3": f"/out/{name}.mp3" if has else None,
            "stale": has and os.path.getmtime(path) > os.path.getmtime(mp3),
            "mtime": os.path.getmtime(mp3) if has else None,
            "modulations": len(spec.get("modulations", [])),
            "tracks": [{
                "name": t.get("name", f"t{i}"),
                "voice": t.get("voice", ""),
                "type": t.get("type", "note"),
                "enter": t.get("enter", 0),
                "exit": t.get("exit", bars),
                "length": t.get("length"),
            } for i, t in enumerate(spec.get("tracks", []))],
            "job": _jobs.get(stem),
        })
    return items


def voices() -> list[dict]:
    """Every voice a spec may name, with where it comes from. Import lazily so an
    engine that fails to import leaves the catalog page up (the error shows here)."""
    try:
        from .drums import DRUM_VOICES
        from .soundbank import SOUNDBANK
        from .synth import VOICES
    except Exception as e:  # noqa: BLE001 - surface the import error in the UI
        return [{"error": str(e)}]
    out = [{"name": n, "group": "synth"} for n in VOICES]
    out += [{"name": n, "group": "drums"} for n in DRUM_VOICES]
    out += [{"name": n, "group": fn.__module__.rsplit(".", 1)[-1]} for n, fn in SOUNDBANK.items()]
    if os.path.isdir(SAMPLES):
        out += [{"name": f"sample:{f[:-4]}", "group": "samples"}
                for f in sorted(os.listdir(SAMPLES)) if f.endswith(".wav")]
    return out


def _voice_mtime(voice: str) -> float:
    """When the voice's source last changed -- a preview older than that is stale."""
    if voice.startswith("sample:"):
        p = os.path.join(SAMPLES, voice.split(":", 1)[1] + ".wav")
        return os.path.getmtime(p) if os.path.exists(p) else 0.0
    from .render import ALL_VOICES
    fn = ALL_VOICES.get(voice)
    mod = sys.modules.get(fn.__module__) if fn else None
    f = getattr(mod, "__file__", None)
    return os.path.getmtime(f) if f and os.path.exists(f) else 0.0


def preview(voice: str, note: int, dur: float = 0.8, tail: float = 0.7) -> str:
    """Render one note of `voice` through the real bus (VOICE_GAIN, pan law) to a
    cached wav under out/preview/. Loudness-matched so the ear compares colour,
    not gain staging (the mix does the staging)."""
    from .render import ALL_VOICES
    if voice.startswith("sample:"):
        if not os.path.exists(os.path.join(SAMPLES, voice.split(":", 1)[1] + ".wav")):
            raise LookupError(f"no sample {voice}")
        dur, tail = 3.0, 0.3  # a recorded phrase needs room to be heard
    elif voice not in ALL_VOICES:
        raise LookupError(f"unknown voice '{voice}'")  # never audition the bell fallback
    os.makedirs(PREVIEW, exist_ok=True)
    key = f"{voice}_{note}".replace(":", "-").replace("/", "-")
    path = os.path.join(PREVIEW, f"{key}.wav")
    if os.path.exists(path) and os.path.getmtime(path) >= _voice_mtime(voice):
        return path
    from .loudness import normalize
    from .render import _bus, write_wav
    from .sequencer import NoteEvent
    from .synth import SR
    ev = NoteEvent(time=0.0, note=note, velocity=100, duration=dur, voice=voice)
    buf = _bus([ev], int(SR * (dur + tail)))
    buf, _ = normalize(buf, PREVIEW_LUFS, -1.0)
    write_wav(path, buf)
    return path


def edit(stem: str, action: str, payload: dict) -> tuple[int, dict]:
    """Splice a track in or out of the spec on disk, then kick a render. Returns
    (http status, body). Validation errors come back as a list, never a 500."""
    from .spec import SpecError
    from .specedit import add_track, remove_track
    path = _spec_path(stem)
    if not path:
        return 404, {"errors": ["no such spec"]}
    with _lock:
        text = open(path).read()
        try:
            if action == "add":
                text, track = add_track(text, payload)
                body = {"ok": True, "track": track}
            elif action == "remove":
                text = remove_track(text, payload.get("name", ""))
                body = {"ok": True}
            else:
                return 404, {"errors": [f"unknown action '{action}'"]}
        except SpecError as e:
            return 400, {"errors": e.errors}
        with open(path, "w") as f:
            f.write(text)
    render(stem)
    return 200, body


def render(stem: str) -> bool:
    path = _spec_path(stem)
    if not path:
        return False
    with _lock:
        if _jobs.get(stem, {}).get("state") == "running":
            return True
        _jobs[stem] = {"state": "running", "log": ""}

    def work():
        p = subprocess.run([sys.executable, os.path.join(ROOT, "cli.py"), "sculpture", path],
                           cwd=ROOT, capture_output=True, text=True)
        _jobs[stem] = {"state": "done" if p.returncode == 0 else "error",
                       "log": (p.stdout + p.stderr).strip()[-2000:]}

    threading.Thread(target=work, daemon=True).start()
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # quiet: audio range requests are chatty
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _file(self, base: str, rel: str, ctype: str):
        path = os.path.abspath(os.path.join(base, rel))
        if not path.startswith(base + os.sep) or not os.path.isfile(path):
            return self._send(404, b"not found", "text/plain")
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        # Range support is what lets the browser seek inside the mp3.
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else size - 1
            end = min(end, size - 1)
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            return self._send(206, data, ctype, {"Accept-Ranges": "bytes",
                                                 "Content-Range": f"bytes {start}-{end}/{size}"})
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype, {"Accept-Ranges": "bytes"})

    def do_GET(self):
        url = unquote(self.path.split("?")[0])
        if url == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url == "/api/catalog":
            return self._send(200, json.dumps(catalog()).encode(), "application/json")
        if url == "/api/voices":
            return self._send(200, json.dumps(voices()).encode(), "application/json")
        if url.startswith("/api/spec/"):
            path = _spec_path(url[len("/api/spec/"):])
            if not path:
                return self._send(404, b"not found", "text/plain")
            return self._send(200, open(path, "rb").read(), "application/json")
        if url.startswith("/api/preview/"):
            q = dict(p.split("=", 1) for p in self.path.partition("?")[2].split("&") if "=" in p)
            try:
                note = max(0, min(127, int(q.get("note", 60))))
                path = preview(url[len("/api/preview/"):], note)
            except LookupError as e:
                return self._send(404, str(e).encode(), "text/plain")
            except Exception as e:  # noqa: BLE001 - a broken voice must not kill the page
                return self._send(500, str(e).encode(), "text/plain")
            return self._file(PREVIEW, os.path.basename(path), "audio/wav")
        if url.startswith("/out/"):
            return self._file(OUT, url[5:], "audio/mpeg")
        if url == "/logo.png":
            return self._file(BRANDING, "unborn-logo-v1.png", "image/png")
        self._send(404, b"not found", "text/plain")

    do_HEAD = do_GET

    def _json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_POST(self):
        url = unquote(self.path)
        if url.startswith("/api/render/"):
            ok = render(url[len("/api/render/"):])
            return self._send(202 if ok else 404, b"{}", "application/json")
        if url.startswith("/api/spec/"):  # /api/spec/<stem>/<add|remove>
            stem, _, action = url[len("/api/spec/"):].partition("/")
            code, body = edit(stem, action, self._json())
            return self._send(code, json.dumps(body).encode(), "application/json")
        self._send(404, b"not found", "text/plain")


def serve(port: int = 8765, open_browser: bool = True) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"  unborn ui -> {url}  (ctrl-c para parar)")
    if open_browser:
        subprocess.Popen(["open", url])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>unborn — sculptures</title>
<link rel="icon" href="/logo.png">
<style>
:root{--bg:#0d0d0f;--panel:#16161a;--line:#26262c;--ink:#e8e6e1;--dim:#8a8890;--acc:#d9b25f;--bad:#d9665f;--mod:#7c8fd9;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:16px 20px;border-bottom:1px solid var(--line)}
header img{width:32px;height:32px;border-radius:6px}header h1{font-size:16px;font-weight:600;margin:0;letter-spacing:.02em}
header span{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:280px 1fr;min-height:calc(100vh - 65px)}
nav{border-right:1px solid var(--line);overflow:auto}
.item{padding:10px 20px;cursor:pointer;border-left:2px solid transparent;display:flex;justify-content:space-between;gap:8px}
.item:hover{background:var(--panel)}.item.on{background:var(--panel);border-left-color:var(--acc)}
.item b{font-weight:500}.item small{color:var(--dim);font-family:var(--mono);font-size:11px;white-space:nowrap}
.tag{font-size:10px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);color:var(--dim)}
.tag.stale{color:var(--acc);border-color:var(--acc)}.tag.none{color:var(--bad);border-color:var(--bad)}
section{padding:24px 28px;overflow:auto}
h2{margin:0 0 4px;font-size:22px;font-weight:600}.meta{color:var(--dim);font-family:var(--mono);font-size:12px}
.bar{display:flex;align-items:center;gap:12px;margin:18px 0}
audio{flex:1;height:36px}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}button:disabled{opacity:.5;cursor:wait}
button.go{border-color:var(--acc);color:var(--acc)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;font:11px var(--mono);color:var(--dim);white-space:pre-wrap;max-height:160px;overflow:auto}
.arr{position:relative;margin-top:12px}
.row{display:grid;grid-template-columns:190px 1fr;align-items:center;height:22px}
.row .lab{font:11px var(--mono);color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:10px;display:flex;gap:6px;align-items:center}
.row .lab i{color:var(--ink);font-style:normal}.row .lab span{flex:1;overflow:hidden;text-overflow:ellipsis}
.row .x{opacity:0;cursor:pointer;color:var(--bad);font-size:13px;line-height:1;padding:0 3px}
.row:hover .x{opacity:1}
.lane{position:relative;height:14px;background:repeating-linear-gradient(90deg,transparent 0 calc(var(--bw) - 1px),var(--line) calc(var(--bw) - 1px) var(--bw))}
.blk{position:absolute;top:2px;bottom:2px;border-radius:3px;background:var(--acc);opacity:.75}
.blk.mod{background:var(--mod)}
.head{position:absolute;top:0;bottom:0;width:1px;background:#fff;pointer-events:none;left:190px;box-shadow:0 0 6px #fff8}
.ruler{display:grid;grid-template-columns:190px 1fr;font:10px var(--mono);color:var(--dim);height:16px}
.ruler div{position:relative}.ruler span{position:absolute;transform:translateX(-50%)}
.lanes{cursor:pointer}
/* add-a-sound panel */
.panel{margin-top:28px;border-top:1px solid var(--line);padding-top:18px}
.ph{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.ph b{font-weight:600}
.ph input{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit;width:160px}
.grp{font:11px var(--mono);color:var(--dim);cursor:pointer;padding:3px 8px;border-radius:9px;border:1px solid transparent}
.grp:hover{color:var(--ink)}.grp.on{border-color:var(--line);color:var(--ink);background:var(--panel)}
.chips{display:flex;flex-wrap:wrap;gap:6px;max-height:150px;overflow:auto;padding:2px}
.chip{font:11px var(--mono);padding:4px 9px;border-radius:12px;border:1px solid var(--line);background:var(--panel);cursor:pointer;color:var(--dim)}
.chip:hover{color:var(--ink);border-color:var(--acc)}.chip.on{color:var(--acc);border-color:var(--acc)}
.chip.play{color:var(--ink);background:var(--acc);border-color:var(--acc)}
.chip.samples{border-style:dashed}
form{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px;margin-top:14px;align-items:end}
label{display:flex;flex-direction:column;gap:3px;font:11px var(--mono);color:var(--dim)}
label input,label select{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:13px var(--mono);width:100%}
label input:focus{outline:none;border-color:var(--acc)}
label.wide{grid-column:span 2}
.acts{grid-column:1/-1;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.acts .hint{font:11px var(--mono);color:var(--dim)}
.err{grid-column:1/-1;color:var(--bad);font:12px var(--mono);white-space:pre-wrap}
.ok{grid-column:1/-1;color:var(--acc);font:12px var(--mono)}
@media(max-width:760px){main{grid-template-columns:1fr}nav{max-height:40vh;border-right:0;border-bottom:1px solid var(--line)}
.row,.ruler{grid-template-columns:110px 1fr}.head{left:110px}.row .x{opacity:1}}
</style></head><body>
<header><img src="/logo.png" alt=""><h1>unborn — sculptures</h1><span>rules in, sound out</span></header>
<main><nav id="list"></nav><section><div id="top"><p class="meta">elige una escultura</p></div><div id="add"></div></section></main>
<script>
let cat=[],cur=null,poll=null,voices=null,panelFor=null,grp='',sel=null;
const pv=new Audio();
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
const fmt=t=>{if(!t)return'';const d=new Date(t*1000);return d.toLocaleDateString('es-MX',{month:'short',day:'numeric'})};
async function load(){cat=await (await fetch('/api/catalog')).json();list();if(cur)view(cat.find(c=>c.stem===cur),false)}
function tag(c){if(c.job&&c.job.state==='running')return'<span class="tag stale">render…</span>';
 if(!c.mp3)return'<span class="tag none">sin mp3</span>';if(c.stale)return'<span class="tag stale">viejo</span>';return''}
function list(){$('#list').innerHTML=cat.map(c=>`<div class="item ${c.stem===cur?'on':''}" data-s="${c.stem}">
 <b>${esc(c.name||c.stem)}</b><small>${tag(c)} ${c.tempo||''}bpm·${c.bars||''}</small></div>`).join('');
 document.querySelectorAll('.item').forEach(e=>e.onclick=()=>{cur=e.dataset.s;list();view(cat.find(c=>c.stem===cur),true);panel(cat.find(c=>c.stem===cur))})}
function view(c,fresh){
 if(!c)return;const a=$('#top audio');const keep=!fresh&&a&&a.dataset.src===c.mp3;
 const running=c.job&&c.job.state==='running';
 const secPerBar=60/c.tempo*c.beats_per_bar,dur=secPerBar*c.bars;
 const bw=`calc(100% / ${c.bars})`;
 const step=c.bars>40?10:c.bars>16?4:2;
 const ticks=[];for(let b=0;b<=c.bars;b+=step)ticks.push(`<span style="left:${b/c.bars*100}%">${b}</span>`);
 const rows=c.tracks.map(t=>`<div class="row"><div class="lab"><span><i>${esc(t.name)}</i> ${esc(t.voice)}</span><a class="x" data-n="${esc(t.name)}" title="quitar '${esc(t.name)}' del spec">×</a></div>
  <div class="lane" style="--bw:${bw}"><div class="blk ${t.type==='modulator'?'mod':''}" style="left:${t.enter/c.bars*100}%;width:${(Math.min(t.exit,c.bars)-t.enter)/c.bars*100}%"
  title="${esc(t.name)} · bars ${t.enter}→${t.exit} · len ${t.length}"></div></div></div>`).join('');
 const html=`<h2>${esc(c.name)}</h2>
 <div class="meta">${c.tempo} bpm · ${c.bars} bars · ${c.tracks.length} tracks · ${c.modulations} mods · ${Math.floor(dur/60)}:${String(Math.round(dur%60)).padStart(2,'0')}${c.mtime?' · render '+fmt(c.mtime):''}</div>
 <div class="bar">${c.mp3?`<audio controls preload="metadata" data-src="${c.mp3}" src="${c.mp3}?v=${c.mtime}"></audio>`:'<span class="meta" style="flex:1">todavía no hay mp3</span>'}
 <button id="rb" ${running?'disabled':''}>${running?'rendereando…':c.mp3?(c.stale?'re-render (spec cambió)':'re-render'):'render'}</button></div>
 ${c.job&&c.job.log?`<pre>${esc(c.job.log)}</pre>`:''}
 <div class="arr"><div class="ruler"><div></div><div>${ticks.join('')}</div></div><div class="lanes">${rows}</div><div class="head" id="ph"></div></div>`;
 if(keep){ // don't reset the player on a status refresh
  $('#rb').outerHTML=html.match(/<button[\s\S]*?<\/button>/)[0];
  $('.lanes').innerHTML=rows;
 }else $('#top').innerHTML=html;
 $('#rb').onclick=async()=>{await fetch('/api/render/'+c.stem,{method:'POST'});watch()};
 document.querySelectorAll('.row .x').forEach(x=>x.onclick=async e=>{e.stopPropagation();const n=x.dataset.n;
  if(!confirm(`quitar el track '${n}' de ${c.stem}.json y re-render?`))return;
  const r=await fetch(`/api/spec/${c.stem}/remove`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:n})});
  const j=await r.json();if(!r.ok)alert((j.errors||[]).join('\n'));await load();watch()});
 const au=$('#top audio'),ph=$('#ph'),lanes=$('.lanes');
 const lane=()=>{const l=document.querySelector('.lane').getBoundingClientRect(),r=$('.arr').getBoundingClientRect();return[l.left-r.left,l.width]};
 const draw=()=>{if(!au||!au.isConnected)return;const[x,w]=lane();ph.style.left=(x+Math.min(1,au.currentTime/(au.duration||dur))*w)+'px';requestAnimationFrame(draw)};
 if(au&&!keep){draw()}
 if(au)lanes.onclick=e=>{if(e.target.classList.contains('x'))return;const[x,w]=lane(),r=$('.arr').getBoundingClientRect();
  const f=(e.clientX-r.left-x)/w;if(f>=0&&f<=1){au.currentTime=f*(au.duration||dur);au.play()}};
 if(running)watch();
}
function watch(){if(poll)return;poll=setInterval(async()=>{await load();
 if(!cat.some(c=>c.job&&c.job.state==='running')){clearInterval(poll);poll=null;const c=cat.find(c=>c.stem===cur);if(c)view(c,true)}},1500)}
/* ---- add a sound ---- */
async function panel(c){
 if(!c||panelFor===c.stem)return;panelFor=c.stem;
 if(!voices)voices=await (await fetch('/api/voices')).json();
 const spec=await (await fetch('/api/spec/'+c.stem)).json();
 const tpb=spec.ticks_per_beat||24,bpb=spec.beats_per_bar||4;
 const groups=[...new Set(voices.map(v=>v.group))];
 $('#add').innerHTML=`<div class="panel">
 <div class="ph"><b>+ sonido</b><input id="q" placeholder="filtrar voces…">
  <span class="grp ${grp===''?'on':''}" data-g="">todas</span>${groups.map(g=>`<span class="grp ${grp===g?'on':''}" data-g="${g}">${g}</span>`).join('')}
  <span class="meta">clic = oír · a la nota del form</span></div>
 <div class="chips" id="chips"></div>
 <form id="tf" autocomplete="off">
  <label class="wide">name<input name="name" placeholder="(= voice)"></label>
  <label class="wide">voice<input name="voice" readonly placeholder="elige una voz arriba"></label>
  <label>note<input name="note" type="number" min="0" max="127" value="60"></label>
  <label>quant<input name="quant" type="number" min="1" value="${tpb/2}" title="ticks por step · tpb=${tpb}: ${tpb/4}=1/16 ${tpb/2}=1/8 ${tpb}=beat ${tpb*bpb}=bar"></label>
  <label>length<input name="length" type="number" min="1" value="7" title="steps en el loop · primos = polymeter"></label>
  <label>pattern<select name="mode"><option value="euclid">euclid</option><option value="steps">steps</option></select></label>
  <label id="l-pulses">pulses<input name="pulses" type="number" min="1" value="3"></label>
  <label id="l-steps" class="wide" style="display:none">steps (vel, coma)<input name="steps" placeholder="90, 0, 70, 0"></label>
  <label>velocity<input name="velocity" type="number" min="1" max="127" value="80"></label>
  <label>enter (bar)<input name="enter" type="number" min="0" max="${c.bars-1}" value="0"></label>
  <label>exit (bar)<input name="exit" type="number" min="1" max="${c.bars}" placeholder="${c.bars}"></label>
  <label>pan<input name="pan" type="number" min="-1" max="1" step="0.1" value="0"></label>
  <label>swing<input name="swing" type="number" min="0" value="0"></label>
  <label>offset<input name="offset" type="number" min="0" value="0"></label>
  <div class="acts"><button type="button" id="hear">▶ oír</button><button type="submit" class="go" id="addb">agregar → render</button>
   <span class="hint">se escribe una línea en specs/${c.stem}.json · validate() decide</span></div>
  <div class="err" id="err"></div><div class="ok" id="okm"></div>
 </form></div>`;
 chips();
 $('#q').oninput=chips;
 document.querySelectorAll('.grp').forEach(g=>g.onclick=()=>{grp=g.dataset.g;document.querySelectorAll('.grp').forEach(x=>x.classList.toggle('on',x.dataset.g===grp));chips()});
 const f=$('#tf');
 f.mode.onchange=()=>{const e=f.mode.value==='euclid';$('#l-pulses').style.display=e?'':'none';$('#l-steps').style.display=e?'none':''};
 $('#hear').onclick=()=>{if(sel)hear(sel)};
 f.onsubmit=async e=>{e.preventDefault();$('#err').textContent='';$('#okm').textContent='';
  if(!sel){$('#err').textContent='elige una voz';return}
  const n=v=>v===''?null:Number(v);
  const t={name:f.name.value.trim()||null,voice:sel,note:n(f.note.value),quant:n(f.quant.value),length:n(f.length.value),
   enter:n(f.enter.value)||null,exit:n(f.exit.value),pan:n(f.pan.value)||null,swing:n(f.swing.value)||null,offset:n(f.offset.value)||null};
  if(f.mode.value==='euclid')t.euclid={pulses:n(f.pulses.value),length:t.length,velocity:n(f.velocity.value)};
  else t.steps=f.steps.value.split(/[\s,]+/).filter(Boolean).map(Number);
  $('#addb').disabled=true;
  const r=await fetch(`/api/spec/${c.stem}/add`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(t)});
  const j=await r.json();$('#addb').disabled=false;
  if(!r.ok){$('#err').textContent=(j.errors||['?']).join('\n');return}
  $('#okm').textContent=`+ ${j.track.name} · rendereando…`;f.name.value='';await load();watch()};
}
function chips(){const q=($('#q')?.value||'').toLowerCase();
 $('#chips').innerHTML=voices.filter(v=>!v.error&&(!grp||v.group===grp)&&v.name.toLowerCase().includes(q))
  .map(v=>`<span class="chip ${v.group} ${v.name===sel?'on':''}" data-v="${esc(v.name)}" title="${v.group}">${esc(v.name)}</span>`).join('')
  ||`<span class="meta">${voices[0]?.error||'nada'}</span>`;
 document.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>{sel=ch.dataset.v;$('#tf').voice.value=sel;
  document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x.dataset.v===sel));hear(sel)})}
function hear(v){const note=$('#tf').note.value||60;pv.pause();
 pv.src=`/api/preview/${encodeURIComponent(v)}?note=${note}`;pv.play().catch(()=>{});
 const ch=[...document.querySelectorAll('.chip')].find(x=>x.dataset.v===v);
 if(ch){ch.classList.add('play');pv.onended=()=>ch.classList.remove('play');pv.onpause=()=>ch.classList.remove('play')}}
document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target===document.body){e.preventDefault();const a=$('#top audio');if(a)a.paused?a.play():a.pause()}});
load();
</script></body></html>"""

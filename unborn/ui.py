"""A local window onto the catalog -- the glass box made visible.

Lists every spec in `specs/`, plays its rendered mp3 from `out/`, re-renders on
demand, and draws the arrangement: each track as a bar over its `enter`/`exit`
window, with a playhead tied to the audio clock. Stdlib only (http.server); a
render runs `cli.py sculpture` in a subprocess so the server never blocks and a
crash in the engine can't take the page down."""
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
        if url.startswith("/out/"):
            return self._file(OUT, url[5:], "audio/mpeg")
        if url == "/logo.png":
            return self._file(BRANDING, "unborn-logo-v1.png", "image/png")
        self._send(404, b"not found", "text/plain")

    do_HEAD = do_GET

    def do_POST(self):
        url = unquote(self.path)
        if url.startswith("/api/render/"):
            ok = render(url[len("/api/render/"):])
            return self._send(202 if ok else 404, b"{}", "application/json")
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
<style>
:root{--bg:#0d0d0f;--panel:#16161a;--line:#26262c;--ink:#e8e6e1;--dim:#8a8890;--acc:#d9b25f;--bad:#d9665f;
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
pre{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;font:11px var(--mono);color:var(--dim);white-space:pre-wrap;max-height:160px;overflow:auto}
.arr{position:relative;margin-top:12px}
.row{display:grid;grid-template-columns:190px 1fr;align-items:center;height:22px}
.row .lab{font:11px var(--mono);color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:10px}
.row .lab i{color:var(--ink);font-style:normal}
.lane{position:relative;height:14px;background:repeating-linear-gradient(90deg,transparent 0 calc(var(--bw) - 1px),var(--line) calc(var(--bw) - 1px) var(--bw))}
.blk{position:absolute;top:2px;bottom:2px;border-radius:3px;background:var(--acc);opacity:.75}
.blk.mod{background:#7c8fd9}
.head{position:absolute;top:0;bottom:0;width:1px;background:#fff;pointer-events:none;left:190px;box-shadow:0 0 6px #fff8}
.ruler{display:grid;grid-template-columns:190px 1fr;font:10px var(--mono);color:var(--dim);height:16px}
.ruler div{position:relative}.ruler span{position:absolute;transform:translateX(-50%)}
.lanes{cursor:pointer}
@media(max-width:760px){main{grid-template-columns:1fr}nav{max-height:40vh;border-right:0;border-bottom:1px solid var(--line)}
.row,.ruler{grid-template-columns:110px 1fr}.head{left:110px}}
</style></head><body>
<header><img src="/logo.png" alt=""><h1>unborn — sculptures</h1><span>rules in, sound out</span></header>
<main><nav id="list"></nav><section id="view"><p class="meta">elige una escultura</p></section></main>
<script>
let cat=[],cur=null,poll=null;
const $=s=>document.querySelector(s);
const fmt=t=>{if(!t)return'';const d=new Date(t*1000);return d.toLocaleDateString('es-MX',{month:'short',day:'numeric'})};
async function load(){cat=await (await fetch('/api/catalog')).json();list();if(cur)view(cat.find(c=>c.stem===cur),false)}
function tag(c){if(c.job&&c.job.state==='running')return'<span class="tag stale">render…</span>';
 if(!c.mp3)return'<span class="tag none">sin mp3</span>';if(c.stale)return'<span class="tag stale">viejo</span>';return''}
function list(){$('#list').innerHTML=cat.map(c=>`<div class="item ${c.stem===cur?'on':''}" data-s="${c.stem}">
 <b>${c.name||c.stem}</b><small>${tag(c)} ${c.tempo||''}bpm·${c.bars||''}</small></div>`).join('');
 document.querySelectorAll('.item').forEach(e=>e.onclick=()=>{cur=e.dataset.s;list();view(cat.find(c=>c.stem===cur),true)})}
function view(c,fresh){
 if(!c)return;const a=$('audio');const keep=!fresh&&a&&a.dataset.src===c.mp3;
 const running=c.job&&c.job.state==='running';
 const secPerBar=60/c.tempo*c.beats_per_bar,dur=secPerBar*c.bars;
 const bw=`calc(100% / ${c.bars})`;
 const step=c.bars>40?10:c.bars>16?4:2;
 const ticks=[];for(let b=0;b<=c.bars;b+=step)ticks.push(`<span style="left:${b/c.bars*100}%">${b}</span>`);
 const rows=c.tracks.map(t=>`<div class="row"><div class="lab"><i>${t.name}</i> ${t.voice}</div>
  <div class="lane" style="--bw:${bw}"><div class="blk ${t.type==='modulator'?'mod':''}" style="left:${t.enter/c.bars*100}%;width:${(Math.min(t.exit,c.bars)-t.enter)/c.bars*100}%"
  title="${t.name} · bars ${t.enter}→${t.exit} · len ${t.length}"></div></div></div>`).join('');
 const html=`<h2>${c.name}</h2>
 <div class="meta">${c.tempo} bpm · ${c.bars} bars · ${c.tracks.length} tracks · ${c.modulations} mods · ${Math.floor(dur/60)}:${String(Math.round(dur%60)).padStart(2,'0')}${c.mtime?' · render '+fmt(c.mtime):''}</div>
 <div class="bar">${c.mp3?`<audio controls preload="metadata" data-src="${c.mp3}" src="${c.mp3}?v=${c.mtime}"></audio>`:'<span class="meta" style="flex:1">todavía no hay mp3</span>'}
 <button id="rb" ${running?'disabled':''}>${running?'rendereando…':c.mp3?(c.stale?'re-render (spec cambió)':'re-render'):'render'}</button></div>
 ${c.job&&c.job.log?`<pre>${c.job.log.replace(/</g,'&lt;')}</pre>`:''}
 <div class="arr"><div class="ruler"><div></div><div>${ticks.join('')}</div></div><div class="lanes">${rows}</div><div class="head" id="ph"></div></div>`;
 if(keep){ // don't reset the player on a status refresh
  $('#rb').outerHTML=html.match(/<button[\s\S]*?<\/button>/)[0];
 }else $('#view').innerHTML=html;
 $('#rb').onclick=async()=>{await fetch('/api/render/'+c.stem,{method:'POST'});watch()};
 const au=$('audio'),ph=$('#ph'),lanes=$('.lanes');
 const lane=()=>{const l=document.querySelector('.lane').getBoundingClientRect(),r=$('.arr').getBoundingClientRect();return[l.left-r.left,l.width]};
 const draw=()=>{if(!au)return;const[x,w]=lane();ph.style.left=(x+Math.min(1,au.currentTime/(au.duration||dur))*w)+'px';requestAnimationFrame(draw)};
 if(au&&!keep){draw();lanes.onclick=e=>{const[x,w]=lane(),r=$('.arr').getBoundingClientRect();
  const f=(e.clientX-r.left-x)/w;if(f>=0&&f<=1){au.currentTime=f*(au.duration||dur);au.play()}}}
 if(running)watch();
}
function watch(){if(poll)return;poll=setInterval(async()=>{await load();
 if(!cat.some(c=>c.job&&c.job.state==='running')){clearInterval(poll);poll=null;const c=cat.find(c=>c.stem===cur);if(c)view(c,true)}},1500)}
document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target===document.body){e.preventDefault();const a=$('audio');if(a)a.paused?a.play():a.pause()}});
load();
</script></body></html>"""

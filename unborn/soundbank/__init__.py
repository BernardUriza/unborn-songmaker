"""Auto-registering sound library. Every public function in a soundbank/*.py
module with the contract `name(freq: float, dur: float) -> np.ndarray` (mono,
float in -1..1, 44100 Hz) is collected into SOUNDBANK and becomes a usable voice
in specs. Codex generates the modules; this loader wires them in with no edits."""
import importlib
import inspect
import pkgutil

SOUNDBANK: dict = {}


def _load() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f".{mod.name}", __name__)
        for name, fn in inspect.getmembers(m, inspect.isfunction):
            if name.startswith("_") or fn.__module__ != m.__name__:
                continue
            SOUNDBANK[name] = fn


_load()

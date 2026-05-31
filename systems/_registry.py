"""Adapter auto-discovery for the Neutral Referee MVP.

`discover()` scans the `systems/` package for every module that exposes BOTH a
module-level `SYSTEM_NAME: str` and a `build_system(config: dict) -> MemorySystem`
callable. Each module is imported inside a try/except so a missing vendor SDK in
one adapter (e.g. `import mem0`) can NEVER break discovery of the others, the
runner, or the web app.

A new adapter author adds exactly ONE file `systems/<x>.py` exposing those two
symbols. They touch no shared registry, no YAML, no `__init__`. This is the
explicit anti-conflict mechanism (DESIGN.md §5.1).

`build_system` may raise `SystemUnavailable` (keys/SDK missing). Discovery does
not call `build_system`, so an import-guarded adapter is still *discovered*
(its name + builder are registered); the unavailability surfaces only when the
runner actually tries to instantiate it, at which point the runner logs a SKIP
and continues.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any, Callable

from systems.base import MemorySystem

BuildFn = Callable[[dict[str, Any]], MemorySystem]

SYSTEMS_DIR = Path(__file__).resolve().parent


class SystemUnavailable(RuntimeError):
    """Raised by an adapter's build_system() when its SDK/key is missing.

    Discovery never treats this as a discovery failure: the adapter is still
    listed (wired), but the runner skips instantiating it.
    """


# Modules in systems/ that are infrastructure, not adapters.
_SKIP_MODULES = {"_registry", "base", "__init__"}


def discover(log: Callable[[str], None] | None = None) -> dict[str, BuildFn]:
    """Return {SYSTEM_NAME: build_system} for every adapter in systems/.

    Import errors (missing SDK) are caught and logged as SKIP lines; they never
    propagate. Returns only modules exposing both required symbols.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    reg: dict[str, BuildFn] = {}
    for mod in pkgutil.iter_modules([str(SYSTEMS_DIR)]):
        if mod.name in _SKIP_MODULES or mod.name.startswith("_"):
            continue
        try:
            m = importlib.import_module(f"systems.{mod.name}")
        except Exception as e:  # import-guarded adapter / missing SDK -> log + skip
            log(f"SKIP discovery of systems.{mod.name}: {type(e).__name__}: {e}")
            continue
        name = getattr(m, "SYSTEM_NAME", None)
        build = getattr(m, "build_system", None)
        if isinstance(name, str) and callable(build):
            reg[name] = build
    return reg


def available_systems(log: Callable[[str], None] | None = None) -> list[str]:
    """Sorted list of discovered system names (for --list / leaderboard)."""
    return sorted(discover(log).keys())

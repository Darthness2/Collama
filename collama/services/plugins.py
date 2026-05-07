"""Plugin loader: discover extra tools from `~/.config/collama/plugins/*.py`.

Each plugin is a Python module that exposes a top-level function `register(reg)`
where `reg` is a callable `register(name, fn, schema)`. Tools added this way
become callable by the model just like the built-ins.

This is a deliberately tiny loader — no sandboxing, no entry-points machinery.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

from ..config import config_dir


Register = Callable[[str, Callable, dict], None]


def plugins_dir() -> Path:
    d = config_dir() / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_all(register: Register) -> list[str]:
    """Import every *.py in the plugins dir and call its register(register)."""
    loaded: list[str] = []
    for f in sorted(plugins_dir().glob("*.py")):
        if f.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"collama_plugin_{f.stem}", f)
        if not spec or not spec.loader:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception as e:
            print(f"plugin {f.name}: {e}")
            continue
        if hasattr(mod, "register"):
            try:
                mod.register(register)
                loaded.append(f.stem)
            except Exception as e:
                print(f"plugin {f.name}.register(): {e}")
    return loaded

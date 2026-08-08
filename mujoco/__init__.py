"""Local MuJoCo tools plus the installed MuJoCo runtime bindings.

This package intentionally shares the official ``mujoco`` package name so
the command-line tools can be run as ``python -m mujoco.<tool>``.  Load the
installed bindings into this package namespace so ``take_screenshots`` keeps
access to ``MjModel``, ``Renderer``, and the rest of the public runtime API.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path


_TOOLS_DIR = Path(__file__).resolve().parent
_RUNTIME_DIR = Path(sysconfig.get_paths()["purelib"]) / "mujoco"
_RUNTIME_INIT = _RUNTIME_DIR / "__init__.py"

if not _RUNTIME_INIT.is_file():
    raise ImportError(
        "The official MuJoCo runtime is required. Install project dependencies with `uv sync`."
    )

# Keep local tools first in __path__, then execute the official package in this
# module's namespace. Its relative imports resolve from the appended runtime
# directory while `python -m mujoco.<tool>` still finds this repository's tools.
__path__.append(str(_RUNTIME_DIR))
__file__ = str(_RUNTIME_INIT)
exec(compile(_RUNTIME_INIT.read_bytes(), str(_RUNTIME_INIT), "exec"), globals())

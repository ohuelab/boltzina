"""Runtime helpers for resolving external command-line tools."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def active_python_bin() -> Path:
    """Return the bin directory for the active Python executable."""
    return Path(sys.executable).parent


def runtime_env(extra_paths: Optional[list[str | Path]] = None) -> dict:
    """Return an environment whose PATH includes runtime tool directories."""
    env = os.environ.copy()
    paths = [str(Path(p)) for p in (extra_paths or [])]
    paths.append(str(active_python_bin()))

    seen = set()
    prefix = []
    for path in paths:
        if path and path not in seen:
            prefix.append(path)
            seen.add(path)

    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(prefix + ([existing] if existing else []))
    return env


def _valid_executable(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and os.access(p, os.X_OK)


def resolve_executable(
    *names: str,
    env_var: Optional[str] = None,
    config_path: Optional[str | Path] = None,
    required: bool = True,
    install_hint: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve an executable from env/config/PATH/active-venv in that order.

    `names` are command names such as "obabel" or "mk_prepare_receptor.py".
    `config_path` is a caller-provided path from Boltzina config, if available.
    """
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value and _valid_executable(env_value):
            return str(Path(env_value))

    if config_path and _valid_executable(config_path):
        return str(Path(config_path))

    for name in names:
        path = shutil.which(name)
        if path:
            return path

    scripts_dir = active_python_bin()
    for name in names:
        path = scripts_dir / name
        if _valid_executable(path):
            return str(path)

    if not required:
        return None

    display_names = ", ".join(names)
    msg = f"Required executable not found: {display_names}."
    if env_var:
        msg += f" You can set {env_var} to an executable path."
    if install_hint:
        msg += f" {install_hint}"
    raise RuntimeError(msg)

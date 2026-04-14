"""
Boltzina tool path management.

Resolves paths for external tools (Vina, MAXIT, Uni-Dock2) in this priority order:
  1. Environment variables (BOLTZINA_VINA_PATH, BOLTZINA_MAXIT_PATH, BOLTZINA_UNIDOCK2_PATH)
  2. ~/.boltzina/config.toml
  3. PATH auto-detection (which vina, etc.)
  4. Error with setup instructions
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

CONFIG_DIR = Path.home() / ".boltzina"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def _load_config() -> dict:
    """Load ~/.boltzina/config.toml, returning empty dict if not found."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    return {}


def _save_config(config: dict) -> None:
    """Save config dict to ~/.boltzina/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(config, f)


def get_vina_path() -> str:
    """Resolve AutoDock Vina binary path."""
    # 1. Environment variable
    env_val = os.environ.get("BOLTZINA_VINA_PATH")
    if env_val and Path(env_val).is_file():
        return env_val

    # 2. config.toml
    cfg = _load_config()
    cfg_val = cfg.get("tools", {}).get("vina")
    if cfg_val and Path(cfg_val).is_file():
        return cfg_val

    # 3. PATH auto-detection
    which_val = shutil.which("vina")
    if which_val:
        return which_val

    raise RuntimeError(
        "AutoDock Vina not found.\n"
        "Install it with: boltzina setup --install-vina\n"
        "Or set BOLTZINA_VINA_PATH environment variable."
    )


def get_maxit_path() -> str:
    """Resolve MAXIT binary path."""
    env_val = os.environ.get("BOLTZINA_MAXIT_PATH")
    if env_val and Path(env_val).is_file():
        return env_val

    cfg = _load_config()
    cfg_val = cfg.get("tools", {}).get("maxit")
    if cfg_val and Path(cfg_val).is_file():
        return cfg_val

    which_val = shutil.which("maxit")
    if which_val:
        return which_val

    raise RuntimeError(
        "MAXIT not found.\n"
        "Install it with: boltzina setup --install-maxit\n"
        "Or set BOLTZINA_MAXIT_PATH environment variable."
    )


def get_unidock2_config() -> dict:
    """
    Resolve Uni-Dock2 binary and environment paths.

    Returns dict with keys: bin, pythonpath, extra_path
    """
    # 1. Environment variable (repo root path)
    env_repo = os.environ.get("BOLTZINA_UNIDOCK2_PATH")
    if env_repo:
        repo = Path(env_repo)
        return _build_unidock2_config_from_repo(repo)

    # 2. config.toml (may store resolved paths directly)
    cfg = _load_config()
    ud2_cfg = cfg.get("tools", {}).get("unidock2_env", {})
    if ud2_cfg.get("bin"):
        return {
            "bin": ud2_cfg["bin"],
            "pythonpath": ud2_cfg.get("pythonpath", ""),
            "extra_path": ud2_cfg.get("extra_path", ""),
        }

    # 3. PATH auto-detection
    which_val = shutil.which("unidock2")
    if which_val:
        return {"bin": which_val, "pythonpath": "", "extra_path": ""}

    raise RuntimeError(
        "Uni-Dock2 not found.\n"
        "Register it with: boltzina setup --register-unidock2 /path/to/Uni-Dock2\n"
        "Or set BOLTZINA_UNIDOCK2_PATH environment variable to the Uni-Dock2 repo root."
    )


def _build_unidock2_config_from_repo(repo: Path) -> dict:
    """Build Uni-Dock2 config from a pixi-based repo root."""
    bin_path = repo / ".pixi" / "envs" / "default" / "bin" / "unidock2"
    if not bin_path.exists():
        raise RuntimeError(
            f"Uni-Dock2 binary not found at {bin_path}.\n"
            f"Have you run 'pixi install' in {repo}?"
        )
    pythonpath = (
        f"{repo}/.pixi/envs/full/lib/python3.10/site-packages"
        f":{repo}/.pixi/envs/default/lib/python3.10/site-packages"
    )
    extra_path = str(repo / ".pixi" / "envs" / "full" / "bin")
    return {
        "bin": str(bin_path),
        "pythonpath": pythonpath,
        "extra_path": extra_path,
    }


def register_vina(path: str) -> None:
    """Register a Vina binary path in config.toml."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Vina binary not found: {path}")
    cfg = _load_config()
    cfg.setdefault("tools", {})["vina"] = str(p.resolve())
    _save_config(cfg)
    print(f"Vina registered: {p.resolve()}")


def register_maxit(path: str) -> None:
    """Register a MAXIT binary path in config.toml."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MAXIT binary not found: {path}")
    cfg = _load_config()
    cfg.setdefault("tools", {})["maxit"] = str(p.resolve())
    _save_config(cfg)
    print(f"MAXIT registered: {p.resolve()}")


def register_unidock2(repo_path: str) -> None:
    """
    Register Uni-Dock2 from its pixi-based repository root.

    Detects binary and PYTHONPATH automatically from the pixi envs.
    """
    repo = Path(repo_path).expanduser().resolve()
    ud2_cfg = _build_unidock2_config_from_repo(repo)
    cfg = _load_config()
    cfg.setdefault("tools", {})["unidock2_env"] = ud2_cfg
    _save_config(cfg)
    print(f"Uni-Dock2 registered from: {repo}")
    print(f"  Binary: {ud2_cfg['bin']}")


def get_boltz_cache() -> Path:
    """Return the Boltz-2 model cache directory."""
    env_val = os.environ.get("BOLTZ_CACHE")
    if env_val:
        return Path(env_val)
    return Path.home() / ".boltz"


def show_config() -> None:
    """Print current tool configuration."""
    cfg = _load_config()
    print(f"Config file: {CONFIG_FILE}")
    tools = cfg.get("tools", {})
    if not tools:
        print("No tools registered. Run 'boltzina setup' to install tools.")
        return

    for key in ("vina", "maxit"):
        val = tools.get(key)
        status = "OK" if val and Path(val).exists() else "NOT FOUND"
        print(f"  {key}: {val or '(not set)'} [{status}]")

    ud2_env = tools.get("unidock2_env", {})
    ud2_bin = ud2_env.get("bin")
    status = "OK" if ud2_bin and Path(ud2_bin).exists() else "NOT FOUND"
    print(f"  unidock2: {ud2_bin or '(not set)'} [{status}]")

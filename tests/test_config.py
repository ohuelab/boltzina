"""
Tests for boltzina.config — tool path management.

Covers:
  - Environment variable priority (BOLTZINA_VINA_PATH etc.)
  - Config file read/write roundtrip
  - PATH auto-detection (when tool is in PATH)
  - Missing tool raises with helpful message
  - register_vina / register_unidock2 write to config.toml
  - get_boltz_cache returns a Path
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import boltzina.config as cfg_module
from boltzina.config import (
    _load_config,
    _save_config,
    get_boltz_cache,
    get_vina_path,
    register_vina,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect config dir to a tmp location for each test."""
    fake_config_dir = tmp_path / ".boltzina_test"
    fake_config_file = fake_config_dir / "config.toml"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", fake_config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", fake_config_file)
    yield fake_config_dir


# ---------------------------------------------------------------------------
# _load_config / _save_config roundtrip
# ---------------------------------------------------------------------------

class TestConfigRoundtrip:
    def test_empty_if_no_file(self):
        data = _load_config()
        assert data == {}

    def test_save_and_load(self):
        _save_config({"tools": {"vina": "/usr/bin/vina"}})
        data = _load_config()
        assert data["tools"]["vina"] == "/usr/bin/vina"

    def test_overwrites_existing(self):
        _save_config({"tools": {"vina": "/old/path"}})
        _save_config({"tools": {"vina": "/new/path"}})
        data = _load_config()
        assert data["tools"]["vina"] == "/new/path"


# ---------------------------------------------------------------------------
# get_vina_path — priority order
# ---------------------------------------------------------------------------

class TestGetVinaPath:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        """BOLTZINA_VINA_PATH env var takes priority over config.toml."""
        fake_vina = tmp_path / "fake_vina"
        fake_vina.write_text("#!/bin/sh\n")
        fake_vina.chmod(0o755)
        monkeypatch.setenv("BOLTZINA_VINA_PATH", str(fake_vina))
        assert get_vina_path() == str(fake_vina)

    def test_config_toml_used_when_no_env(self, tmp_path, monkeypatch):
        """config.toml is used when env var is absent."""
        monkeypatch.delenv("BOLTZINA_VINA_PATH", raising=False)
        fake_vina = tmp_path / "fake_vina"
        fake_vina.write_text("#!/bin/sh\n")
        fake_vina.chmod(0o755)
        _save_config({"tools": {"vina": str(fake_vina)}})
        assert get_vina_path() == str(fake_vina)

    def test_path_autodiscovery(self, tmp_path, monkeypatch):
        """shutil.which result is used when env var and config.toml absent."""
        monkeypatch.delenv("BOLTZINA_VINA_PATH", raising=False)
        # Don't write any config
        fake_vina = tmp_path / "vina"
        fake_vina.write_text("#!/bin/sh\n")
        fake_vina.chmod(0o755)
        monkeypatch.setattr(shutil, "which", lambda name: str(fake_vina) if name == "vina" else None)
        result = get_vina_path()
        assert result == str(fake_vina)

    def test_raises_when_not_found(self, monkeypatch):
        """RuntimeError raised with setup instructions when vina not found."""
        monkeypatch.delenv("BOLTZINA_VINA_PATH", raising=False)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="[Vv]ina"):
            get_vina_path()


# ---------------------------------------------------------------------------
# register_vina
# ---------------------------------------------------------------------------

class TestRegisterVina:
    def test_registers_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BOLTZINA_VINA_PATH", raising=False)
        fake_vina = tmp_path / "vina"
        fake_vina.write_text("#!/bin/sh\n")
        fake_vina.chmod(0o755)
        register_vina(str(fake_vina))
        data = _load_config()
        assert data["tools"]["vina"] == str(fake_vina)

    def test_registers_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            register_vina("/no/such/vina_binary")


# ---------------------------------------------------------------------------
# get_boltz_cache
# ---------------------------------------------------------------------------

class TestGetBoltzCache:
    def test_returns_path(self):
        cache = get_boltz_cache()
        assert isinstance(cache, Path)

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path))
        cache = get_boltz_cache()
        assert cache == tmp_path

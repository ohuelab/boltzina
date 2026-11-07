#!/bin/sh
set -eu

if [ "$(uname -m)" != x86_64 ]; then
    echo "This image recipe requires x86_64 for the bundled AutoDock Vina binary." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential python3-dev bison flex csh wget ca-certificates libgomp1 zlib1g-dev \
    libxrender1 libxext6 libsm6
apt-get clean
python -m pip install --no-cache-dir uv==0.6.16
cd /opt/boltzina
python - <<'PY'
from pathlib import Path
import shutil
for cache in Path("boltzina").rglob("__pycache__"):
    shutil.rmtree(cache)
PY
uv sync --project environments/bioir --locked --no-dev
uv cache clean
environments/bioir/.venv/bin/python - <<'PY'
from pathlib import Path
from boltzina import config
config.CONFIG_DIR = Path("/opt/boltzina-tools")
config.CONFIG_FILE = config.CONFIG_DIR / "config.toml"
from boltzina.cli import _install_maxit, _install_vina
_install_vina()
_install_maxit()
PY
mkdir -p /cache/boltz /data

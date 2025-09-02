#!/usr/bin/env python3
from boltz.main import download_boltz1, download_boltz2
from pathlib import Path
cache = "~/.boltz"
cache = Path(cache).expanduser()
cache.mkdir(parents=True, exist_ok=True)
download_boltz1(cache)
download_boltz2(cache)

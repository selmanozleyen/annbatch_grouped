"""Load DATA_DIR / RESULTS_DIR from the project-root paths.conf file.

Resolution order for each key:
  1. Environment variable  (e.g. DATA_DIR=/some/path python ...)
  2. paths.conf in the repo root
  3. Hard-coded fallback (./data, ./results)

The conf file is intentionally gitignored so every user sets their own paths.
"""

from __future__ import annotations

import os
from pathlib import Path

_CONF_NAME = "paths.conf"
_DEFAULTS = {
    "DATA_DIR": "./data",
    "RESULTS_DIR": "./results",
    "ANNBATCH_REPO": "https://github.com/scverse/annbatch.git",
    "ANNBATCH_REF": "",
    "TAHOE_PATH": "",
}


def _find_conf() -> Path | None:
    """Walk up from this file to find paths.conf in the repo root."""
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = cur / _CONF_NAME
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _parse_conf(path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split("#")[0].strip()
        kv[key.strip()] = value
    return kv


def _load() -> dict[str, str]:
    conf = _find_conf()
    file_vals = _parse_conf(conf) if conf else {}
    result: dict[str, str] = {}
    for key, fallback in _DEFAULTS.items():
        result[key] = os.environ.get(key) or file_vals.get(key) or fallback
    return result


_cfg = _load()

DATA_DIR: Path = Path(_cfg["DATA_DIR"])
RESULTS_DIR: Path = Path(_cfg["RESULTS_DIR"])
ANNBATCH_REPO: str = _cfg["ANNBATCH_REPO"]
ANNBATCH_REF: str = _cfg["ANNBATCH_REF"]
TAHOE_PATH: str = _cfg["TAHOE_PATH"]

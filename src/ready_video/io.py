from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    if mode is not None:
        os.chmod(tmp, mode)
    tmp.replace(path)


def atomic_write_json(path: Path, data: Any, mode: int | None = None) -> None:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n", mode=mode)


def atomic_write_yaml(path: Path, data: Any) -> None:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))

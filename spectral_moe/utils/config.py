from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(config_path.resolve())
    data["_config_dir"] = str(config_path.resolve().parent)
    return data

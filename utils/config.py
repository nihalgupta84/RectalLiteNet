"""Configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration and verify its main sections."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required_sections = {"data", "model", "training", "inference"}
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(
            f"Configuration {config_path} is missing sections: {sorted(missing_sections)}"
        )
    return config

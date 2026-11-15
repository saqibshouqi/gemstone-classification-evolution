"""Load the shared YAML experiment configuration."""

from pathlib import Path
from typing import Any

import yaml

from gemstone_trends.paths import resolve_path


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = resolve_path(path or Path("configs/default.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config

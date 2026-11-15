"""Project-root helpers used by command-line entry points."""

from pathlib import Path


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project root (pyproject.toml).")


def resolve_path(path: Path | str, *, root: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (root or project_root()) / value

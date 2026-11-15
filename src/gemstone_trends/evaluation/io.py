"""Read and validate the shared prediction CSV contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath

import numpy as np

from gemstone_trends.config import load_config
from gemstone_trends.data.splits import SPLIT_COLUMNS, project_root, read_lines


PREDICTION_PREFIX = ("path", "true", "where")


def normalize_path(value: str) -> str:
    text = value.strip().replace("\\", "/")
    if not text:
        return ""
    while text.startswith("./"):
        text = text[2:]
    return PurePosixPath(text).as_posix()


def read_split_contract(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
            raise ValueError(f"{path} must have columns: {','.join(SPLIT_COLUMNS)}")
        rows = list(reader)
    by_path = {normalize_path(row["path"]): row for row in rows}
    if len(by_path) != len(rows):
        raise ValueError(f"The split file contains duplicate image paths: {path}")
    return rows, by_path


def read_predictions(path: Path, classes: list[str]) -> list[dict[str, object]]:
    expected_fields = (*PREDICTION_PREFIX, *classes)
    parsed: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != expected_fields:
            raise ValueError(
                f"{path.name}: expected columns path,true,where followed by all {len(classes)} "
                "class probabilities in splits/classes.txt order."
            )
        for row_number, raw in enumerate(reader, start=2):
            image_path = normalize_path(raw["path"] or "")
            label = (raw["true"] or "").strip()
            where = (raw["where"] or "").strip().lower()
            if not image_path:
                raise ValueError(f"{path.name}:{row_number}: path is empty.")
            if label not in classes:
                raise ValueError(f"{path.name}:{row_number}: unknown true class {label!r}.")
            if where not in {"test", "oof"}:
                raise ValueError(f"{path.name}:{row_number}: where must be 'test' or 'oof'.")
            key = (image_path, where)
            if key in seen:
                raise ValueError(f"{path.name}:{row_number}: duplicate path/where pair {key}.")
            seen.add(key)
            try:
                probabilities = np.asarray([float(raw[name]) for name in classes], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{row_number}: invalid probability value.") from exc
            if not np.isfinite(probabilities).all():
                raise ValueError(f"{path.name}:{row_number}: probabilities must be finite.")
            if np.any(probabilities < -1e-8) or np.any(probabilities > 1 + 1e-8):
                raise ValueError(f"{path.name}:{row_number}: probabilities must be in [0, 1].")
            if not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-3):
                raise ValueError(
                    f"{path.name}:{row_number}: probability row sums to "
                    f"{probabilities.sum():.6f}, not 1."
                )
            parsed.append({
                "path": image_path,
                "true": label,
                "where": where,
                "probabilities": probabilities,
            })
    if not parsed:
        raise ValueError(f"Prediction file is empty: {path}")
    return parsed


def validate_predictions(
    prediction_path: Path,
    split_rows: list[dict[str, str]],
    split_by_path: dict[str, dict[str, str]],
    classes: list[str],
    *,
    allow_partial: bool = False,
) -> list[dict[str, object]]:
    predictions = read_predictions(prediction_path, classes)
    covered: dict[str, set[str]] = {"test": set(), "oof": set()}
    expected: dict[str, set[str]] = {
        "test": {normalize_path(row["path"]) for row in split_rows if row["split"] == "test"},
        "oof": {normalize_path(row["path"]) for row in split_rows if row["split"] == "train"},
    }
    for row in predictions:
        image_path = str(row["path"])
        split_row = split_by_path.get(image_path)
        if split_row is None:
            raise ValueError(f"{prediction_path.name}: path is absent from the canonical split: {image_path}")
        target_split = "test" if row["where"] == "test" else "train"
        if split_row["split"] != target_split:
            raise ValueError(
                f"{prediction_path.name}: {row['where']} prediction points to a "
                f"{split_row['split']} row: {image_path}"
            )
        if split_row["class"] != row["true"]:
            raise ValueError(
                f"{prediction_path.name}: true label {row['true']!r} does not match "
                f"split label {split_row['class']!r} for {image_path}"
            )
        covered[str(row["where"])].add(image_path)
    if not allow_partial:
        for where, actual_paths in covered.items():
            if actual_paths and actual_paths != expected[where]:
                missing = len(expected[where] - actual_paths)
                unexpected = len(actual_paths - expected[where])
                raise ValueError(
                    f"{prediction_path.name}: incomplete {where} coverage "
                    f"(missing={missing}, unexpected={unexpected})."
                )
    return predictions


def _prediction_files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    if not found:
        raise FileNotFoundError("No prediction CSV files were found.")
    return found


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    config = load_config()
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--classes-file", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Allow a handshake/sample file.")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project / path
    split_rows, split_by_path = read_split_contract(
        resolve(args.split_file or Path(data_config["split_file"]))
    )
    classes = read_lines(resolve(args.classes_file or Path(data_config["classes_file"])))
    paths = [resolve(path) for path in (args.paths or [Path(evaluation_config["prediction_directory"])])]
    prediction_files = _prediction_files(paths)
    for path in prediction_files:
        validate_predictions(
            path, split_rows, split_by_path, classes, allow_partial=args.allow_partial
        )
        print(f"OK {path.relative_to(project).as_posix()}: prediction contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

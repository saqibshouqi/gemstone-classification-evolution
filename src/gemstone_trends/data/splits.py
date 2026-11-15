"""Build and validate the fixed Chow 68-class image split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from gemstone_trends.config import load_config
from gemstone_trends.paths import project_root


SPLIT_COLUMNS = ("path", "class", "split", "fold", "dup_group")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_manifest(path: Path, class_order: list[str]) -> list[tuple[str, str]]:
    """Return ``(class_name, filename)`` pairs from Class/filename manifests."""
    class_set = set(class_order)
    selected: list[tuple[str, str]] = []
    for line_number, raw in enumerate(read_lines(path), start=1):
        relative = raw.replace("\\", "/")
        parts = relative.split("/", maxsplit=1)
        if len(parts) != 2 or parts[0] not in class_set or not parts[1]:
            raise ValueError(f"Invalid manifest entry at {path}:{line_number}: {raw!r}")
        selected.append((parts[0], parts[1]))
    if len({(label, filename.casefold()) for label, filename in selected}) != len(selected):
        raise ValueError(f"Manifest contains duplicate class/file entries: {path}")
    return selected


def _relative_project_path(image_path: Path, root: Path) -> str:
    try:
        return image_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Dataset image {image_path} is outside the project root {root}. "
            "Place the dataset inside the workspace or update the path contract."
        ) from exc


def _collect_split(
    data_root: Path,
    split: str,
    manifest: Path,
    class_order: list[str],
    root: Path,
) -> list[dict[str, str]]:
    selected = read_manifest(manifest, class_order)
    seen_paths: set[str] = set()
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for label, filename in selected:
        image_path = data_root / split / label / Path(filename)
        if not image_path.is_file():
            missing.append(image_path.as_posix())
            continue
        relative_path = _relative_project_path(image_path, root)
        if relative_path in seen_paths:
            raise ValueError(f"Selected image appears more than once: {relative_path}")
        seen_paths.add(relative_path)
        rows.append({"path": relative_path, "class": label, "split": split})
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        suffix = "\n  ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} selected {split} image(s) are missing under {data_root}:\n"
            f"{preview}{suffix}"
        )
    return rows


def _initial_group(path: str) -> str:
    return "dup-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]


def build_rows(
    *,
    root: Path,
    data_root: Path,
    class_file: Path,
    train_manifest: Path,
    test_manifest: Path,
    folds: int = 5,
    seed: int = 42,
    expected_classes: int = 68,
    expected_train: int = 2042,
    expected_test: int = 284,
) -> list[dict[str, str]]:
    classes = read_lines(class_file)
    if len(classes) != expected_classes or len(set(classes)) != len(classes):
        raise ValueError(
            f"Expected {expected_classes} unique classes in {class_file}; found {len(classes)}."
        )
    if folds < 2:
        raise ValueError("At least two stratified folds are required.")

    train_rows = _collect_split(data_root, "train", train_manifest, classes, root)
    test_rows = _collect_split(data_root, "test", test_manifest, classes, root)
    if len(train_rows) != expected_train or len(test_rows) != expected_test:
        raise ValueError(
            "The source manifests do not match the canonical benchmark counts: "
            f"train={len(train_rows)} (expected {expected_train}), "
            f"test={len(test_rows)} (expected {expected_test})."
        )

    class_to_index = {name: index for index, name in enumerate(classes)}
    train_rows.sort(key=lambda row: (class_to_index[row["class"]], row["path"].casefold()))
    test_rows.sort(key=lambda row: (class_to_index[row["class"]], row["path"].casefold()))

    counts = Counter(row["class"] for row in train_rows)
    if min(counts.values(), default=0) < folds:
        raise ValueError("Each training class must have at least one image per CV fold.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [row["class"] for row in train_rows]
    for fold_index, (_, validation_indices) in enumerate(splitter.split([0] * len(labels), labels)):
        for row_index in validation_indices:
            train_rows[row_index]["fold"] = str(fold_index)

    for row in train_rows:
        row["dup_group"] = _initial_group(row["path"])
    for row in test_rows:
        row["fold"] = ""
        row["dup_group"] = _initial_group(row["path"])

    return train_rows + test_rows


def write_split(rows: list[dict[str, str]], output: Path, *, overwrite: bool = False) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace the fixed split at {output}. "
            "Pass --overwrite only when intentionally rebuilding it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def write_excluded_class_report(data_root: Path, class_file: Path, output: Path) -> int:
    """Record source classes omitted from the fixed 68-class benchmark."""
    selected = set(read_lines(class_file))
    train_root = data_root / "train"
    test_root = data_root / "test"
    source_classes = sorted(
        (directory.name for directory in train_root.iterdir() if directory.is_dir()),
        key=str.casefold,
    )
    excluded = [name for name in source_classes if name not in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class", "source_train_images", "source_test_images", "documented_reason"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in excluded:
            if label in {"Garnet", "Garnet Red"}:
                reason = (
                    "Removed in the 2022 paper because garnet labels overlap several other classes; "
                    "the raw Kaggle folder is named Garnet Red."
                )
            elif label == "Moonstone":
                reason = "Removed in the 2022 paper because its images mix orange, white, and yellow stones."
            else:
                reason = (
                    "Outside the selected 68 classes; the paper reports excluding classes with fewer "
                    "than 24 usable training images, but does not map that reason to each class here."
                )
            writer.writerow({
                "class": label,
                "source_train_images": sum(1 for path in (train_root / label).iterdir() if path.is_file()),
                "source_test_images": sum(
                    1 for path in (test_root / label).iterdir() if path.is_file()
                ) if (test_root / label).is_dir() else 0,
                "documented_reason": reason,
            })
    os.replace(temporary, output)
    return len(excluded)


def write_excluded_training_image_report(
    *,
    root: Path,
    data_root: Path,
    class_file: Path,
    train_manifest: Path,
    output: Path,
) -> int:
    """List retained-class source files absent from the paper's train selection."""
    classes = read_lines(class_file)
    selected = set(read_manifest(train_manifest, classes))
    removed: list[dict[str, str]] = []
    for label in classes:
        class_dir = data_root / "train" / label
        for image_path in sorted(class_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if (label, image_path.name) not in selected:
                removed.append({
                    "path": _relative_project_path(image_path, root),
                    "class": label,
                    "split": "train",
                    "documented_reason": (
                        "Not present in the reference train_selected manifest; the paper's "
                        "cleaning discarded poorly segmented training images, but does not "
                        "document a per-file reason here."
                    ),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "class", "split", "documented_reason"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(removed)
    os.replace(temporary, output)
    return len(removed)


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    config = load_config()
    data_config = config["data"]
    project_config = config["project"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--classes", type=Path)
    parser.add_argument(
        "--train-manifest", type=Path,
    )
    parser.add_argument(
        "--test-manifest", type=Path,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--excluded-classes-output", type=Path,
    )
    parser.add_argument("--excluded-training-images-output", type=Path)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    project = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project / path
    data_root = resolve(args.data_root or Path(data_config["root"]))
    class_file = resolve(args.classes or Path(data_config["classes_file"]))
    train_manifest = resolve(args.train_manifest or Path(data_config["train_manifest"]))
    test_manifest = resolve(args.test_manifest or Path(data_config["test_manifest"]))
    output = resolve(args.output or Path(data_config["split_file"]))
    excluded_output = resolve(
        args.excluded_classes_output or Path("splits/excluded_classes.csv")
    )
    folds = args.folds or int(data_config["folds"])
    seed = args.seed if args.seed is not None else int(project_config["seed"])
    rows = build_rows(
        root=project,
        data_root=data_root,
        class_file=class_file,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        folds=folds,
        seed=seed,
        expected_classes=int(data_config["expected_classes"]),
        expected_train=int(data_config["expected_train_images"]),
        expected_test=int(data_config["expected_test_images"]),
    )
    write_split(rows, output, overwrite=args.overwrite)
    excluded_count = write_excluded_class_report(
        data_root, class_file, excluded_output
    )
    excluded_training_count = write_excluded_training_image_report(
        root=project,
        data_root=data_root,
        class_file=class_file,
        train_manifest=train_manifest,
        output=resolve(
            args.excluded_training_images_output
            or Path("splits/excluded_training_images.csv")
        ),
    )
    train_count = sum(row["split"] == "train" for row in rows)
    test_count = len(rows) - train_count
    print(
        f"Wrote {output.relative_to(project).as_posix()}: {train_count} train, "
        f"{test_count} test images, 68 classes, {folds} train folds."
    )
    print("Test rows have no CV fold; duplicate groups are provisional until the audit runs.")
    print(f"Recorded {excluded_count} source classes outside the fixed benchmark.")
    print(f"Recorded {excluded_training_count} raw training images omitted within retained classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

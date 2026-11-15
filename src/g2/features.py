"""Versioned, cached feature extraction for the G2 anchor."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from gemstone_trends.config import load_config
from gemstone_trends.data.splits import read_lines
from gemstone_trends.evaluation.io import read_split_contract
from gemstone_trends.paths import project_root
from g2.segmentation import MASK_MODES, get_or_create_mask, rgb_to_grey


FEATURE_NAME = "rgb8_lbp8_r1"
FEATURE_SCHEMA = "g2-rgb8-joint-lbp8-radius1-v1"
HISTOGRAM_LENGTH = 8 ** 3
LBP_LENGTH = 2 ** 8
FEATURE_LENGTH = HISTOGRAM_LENGTH + LBP_LENGTH
FEATURE_VIEWS = ("rgb8", "lbp8_r1", FEATURE_NAME)


def feature_view_length(name: str) -> int:
    """Return the fixed width of an E3 G2 feature view."""
    if name == "rgb8":
        return HISTOGRAM_LENGTH
    if name == "lbp8_r1":
        return LBP_LENGTH
    if name == FEATURE_NAME:
        return FEATURE_LENGTH
    raise ValueError(f"Unknown G2 feature view {name!r}; choose from {', '.join(FEATURE_VIEWS)}.")


def select_feature_view(matrix: np.ndarray, name: str) -> np.ndarray:
    """Select histogram-only, LBP-only, or concatenated features from cache."""
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] != FEATURE_LENGTH:
        raise ValueError(f"Expected a matrix with {FEATURE_LENGTH} cached G2 features.")
    feature_view_length(name)
    if name == "rgb8":
        return values[:, :HISTOGRAM_LENGTH]
    if name == "lbp8_r1":
        return values[:, HISTOGRAM_LENGTH:]
    return values


@dataclass(frozen=True)
class FeatureDataset:
    x: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    splits: np.ndarray
    folds: np.ndarray
    classes: tuple[str, ...]
    mode: str
    fingerprint: str

    @property
    def feature_length(self) -> int:
        return int(self.x.shape[1])


def _rgb_histogram(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if pixels.size == 0:
        return np.zeros(bins ** 3, dtype=np.float32)
    histogram, _ = np.histogramdd(
        pixels.astype(np.float32),
        bins=(bins, bins, bins),
        range=((0.0, 256.0), (0.0, 256.0), (0.0, 256.0)),
    )
    histogram = histogram.astype(np.float32, copy=False)
    total = float(histogram.sum())
    if total:
        histogram /= total
    return histogram.ravel(order="C")


def _circular_lbp_codes(grey: np.ndarray, points: int = 8, radius: float = 1.0) -> np.ndarray:
    """Circular LBP with bilinear sampling and edge replication."""
    height, width = grey.shape
    padded = np.pad(np.asarray(grey, dtype=np.float32), 2, mode="edge")
    row_base = np.arange(height, dtype=np.float32) + 2.0
    col_base = np.arange(width, dtype=np.float32) + 2.0
    codes = np.zeros((height, width), dtype=np.uint16)
    epsilon = np.finfo(np.float32).eps * 4
    for point in range(points):
        angle = 2.0 * np.pi * point / points
        row = row_base - radius * np.sin(angle)
        col = col_base + radius * np.cos(angle)
        row0 = np.floor(row).astype(np.intp)
        col0 = np.floor(col).astype(np.intp)
        row1 = row0 + 1
        col1 = col0 + 1
        dy = (row - row0).astype(np.float32)[:, None]
        dx = (col - col0).astype(np.float32)[None, :]
        top_left = padded[row0[:, None], col0[None, :]]
        top_right = padded[row0[:, None], col1[None, :]]
        bottom_left = padded[row1[:, None], col0[None, :]]
        bottom_right = padded[row1[:, None], col1[None, :]]
        top = top_left * (1.0 - dx) + top_right * dx
        bottom = bottom_left * (1.0 - dx) + bottom_right * dx
        neighbour = top * (1.0 - dy) + bottom * dy
        codes |= ((neighbour >= grey - epsilon).astype(np.uint16) << point)
    return codes


def _lbp_histogram(rgb: np.ndarray, points: int = 8, radius: float = 1.0) -> np.ndarray:
    # LBP covers the complete image, including segmented background. The RGB
    # histogram alone is limited to foreground pixels.
    grey = rgb_to_grey(rgb) / 255.0
    codes = _circular_lbp_codes(grey, points=points, radius=radius)
    histogram = np.bincount(codes.ravel(), minlength=2 ** points).astype(np.float32)
    total = float(histogram.sum())
    if total:
        histogram /= total
    return histogram


def extract_features(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return 512 normalized joint RGB bins plus 256 normalized LBP bins."""
    image = np.asarray(rgb, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Feature extraction expects an RGB image.")
    if binary.shape != image.shape[:2]:
        raise ValueError("Segmentation mask dimensions must match the source image.")
    segmented = np.where(binary[..., None], image, 0).astype(np.uint8, copy=False)
    vector = np.concatenate((_rgb_histogram(image, binary), _lbp_histogram(segmented)))
    if vector.shape != (FEATURE_LENGTH,):
        raise RuntimeError(f"Feature schema produced {vector.size} values, expected {FEATURE_LENGTH}.")
    return vector.astype(np.float32, copy=False)


def _fingerprint(project: Path, split_file: Path, rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(split_file.read_bytes())
    for row in rows:
        relative = row["path"].replace("\\", "/")
        source = project / Path(relative)
        if not source.is_file():
            raise FileNotFoundError(f"Canonical image is missing: {source}")
        stat = source.stat()
        record = (
            f"\n{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0{row['class']}\0"
            f"{row['split']}\0{row['fold']}"
        )
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def _cache_path(project: Path, mode: str) -> Path:
    return project / "data" / "processed" / "g2" / f"{mode}_{FEATURE_NAME}.npz"


def _load_cache(
    path: Path, *, fingerprint: str, mode: str, classes: tuple[str, ...]
) -> FeatureDataset | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            if (
                metadata.get("schema") != FEATURE_SCHEMA
                or metadata.get("feature_name") != FEATURE_NAME
                or metadata.get("fingerprint") != fingerprint
                or metadata.get("mode") != mode
                or tuple(stored["classes"].astype(str).tolist()) != classes
            ):
                return None
            x = stored["x"].astype(np.float32, copy=True)
            labels = stored["labels"].astype(str, copy=True)
            paths = stored["paths"].astype(str, copy=True)
            splits = stored["splits"].astype(str, copy=True)
            folds = stored["folds"].astype(np.int32, copy=True)
        if (
            x.ndim != 2 or x.shape[1] != FEATURE_LENGTH
            or any(len(array) != len(x) for array in (labels, paths, splits, folds))
        ):
            return None
        return FeatureDataset(x, labels, paths, splits, folds, classes, mode, fingerprint)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, dataset: FeatureDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": FEATURE_SCHEMA,
        "feature_name": FEATURE_NAME,
        "feature_length": dataset.feature_length,
        "mode": dataset.mode,
        "fingerprint": dataset.fingerprint,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            x=dataset.x.astype(np.float32, copy=False),
            labels=dataset.labels.astype(str),
            paths=dataset.paths.astype(str),
            splits=dataset.splits.astype(str),
            folds=dataset.folds.astype(np.int32, copy=False),
            classes=np.asarray(dataset.classes, dtype=str),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, path)


def load_feature_dataset(
    *,
    project: Path | None = None,
    mode: str = "labelfree",
    refresh: bool = False,
    cache_path: Path | None = None,
) -> FeatureDataset:
    """Load a valid cache or extract the canonical train and test features."""
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown G2 segmentation mode: {mode}")
    root = (project or project_root()).resolve()
    config = load_config(root / "configs" / "default.yaml")
    data_config = config["data"]
    g2_config = config.get("g2", {})
    if (
        g2_config.get("feature_name", FEATURE_NAME) != FEATURE_NAME
        or int(g2_config.get("feature_length", FEATURE_LENGTH)) != FEATURE_LENGTH
    ):
        raise ValueError("G2 feature config does not match this implementation's feature schema.")
    split_file = Path(data_config["split_file"])
    split_file = split_file if split_file.is_absolute() else root / split_file
    classes_file = Path(data_config["classes_file"])
    classes_file = classes_file if classes_file.is_absolute() else root / classes_file
    rows, _ = read_split_contract(split_file)
    classes = tuple(read_lines(classes_file))
    if len(classes) != 68 or len(set(classes)) != 68:
        raise ValueError("G2 requires the canonical ordered list of 68 classes.")
    expected_train = int(data_config["expected_train_images"])
    expected_test = int(data_config["expected_test_images"])
    actual_train = sum(row["split"] == "train" for row in rows)
    actual_test = sum(row["split"] == "test" for row in rows)
    if (actual_train, actual_test) != (expected_train, expected_test):
        raise ValueError(
            f"Canonical split counts changed: train={actual_train}, test={actual_test}; "
            f"expected train={expected_train}, test={expected_test}."
        )
    fingerprint = _fingerprint(root, split_file, rows)
    target = cache_path or _cache_path(root, mode)
    if not target.is_absolute():
        target = root / target
    if not refresh:
        cached = _load_cache(target, fingerprint=fingerprint, mode=mode, classes=classes)
        if cached is not None:
            return cached

    max_label = max(len(row["class"]) for row in rows)
    max_path = max(len(row["path"]) for row in rows)
    matrix = np.empty((len(rows), FEATURE_LENGTH), dtype=np.float32)
    labels = np.empty(len(rows), dtype=f"<U{max_label}")
    paths = np.empty(len(rows), dtype=f"<U{max_path}")
    splits = np.empty(len(rows), dtype="<U5")
    folds = np.full(len(rows), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        relative = row["path"].replace("\\", "/")
        source = root / Path(relative)
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask, _variant = get_or_create_mask(root, row, rgb, mode, refresh=refresh)
        matrix[index] = extract_features(rgb, mask)
        labels[index] = row["class"]
        paths[index] = relative
        splits[index] = row["split"]
        if row["split"] == "train":
            folds[index] = int(row["fold"])
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"Extracted {index + 1}/{len(rows)} {mode} feature rows.")
    dataset = FeatureDataset(
        matrix, labels, paths, splits, folds, classes, mode, fingerprint
    )
    _write_cache(target, dataset)
    print(f"Cached {len(rows)} rows in {target.relative_to(root).as_posix()}.")
    return dataset

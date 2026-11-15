"""Otsu gemstone masks used by the G2 (2022) reproduction anchor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from scipy import ndimage

from gemstone_trends.config import load_config
from gemstone_trends.data.splits import read_lines
from gemstone_trends.evaluation.io import read_split_contract
from gemstone_trends.paths import project_root


SEGMENTATION_SCHEMA = "g2-otsu-v1"
MASK_MODES = ("published", "labelfree", "saturation", "grey")
GREY_CLASSES = frozenset({
    "Alexandrite", "Almandine", "Aquamarine", "Bloodstone",
    "Blue Lace Agate", "Chrome Diopside", "Diamond", "Fluorite",
    "Onyx Black", "Pyrope", "Quartz Beer", "Quartz Rutilated",
    "Rhodolite", "Rhodonite", "Sapphire Blue", "Sodalite",
})


def rgb_to_grey(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to the conventional 0..255 luminance plane."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    values = image.astype(np.float32, copy=False)
    return 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]


def rgb_to_saturation(rgb: np.ndarray) -> np.ndarray:
    """Return HSV saturation scaled to 0..255 without an optional CV dependency."""
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    high = values.max(axis=2)
    low = values.min(axis=2)
    delta = high - low
    saturation = np.divide(
        delta * 255.0,
        high,
        out=np.zeros_like(high),
        where=high > 0,
    )
    return saturation.astype(np.float32, copy=False)


def _otsu_threshold(plane: np.ndarray) -> float:
    """Compute Otsu's between-class variance threshold over 256 fixed bins."""
    values = np.asarray(plane, dtype=np.float32)
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 256.0))
    total = int(histogram.sum())
    if total == 0:
        return 0.0
    probabilities = histogram.astype(np.float64) / total
    centers = (edges[:-1] + edges[1:]) * 0.5
    background_weight = np.cumsum(probabilities)
    background_mean = np.cumsum(probabilities * centers)
    foreground_weight = 1.0 - background_weight
    denominator = background_weight * foreground_weight
    numerator = (background_mean[-1] * background_weight - background_mean) ** 2
    variance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return float(centers[int(np.argmax(variance))])


def _disk(radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def _corner_mean(plane: np.ndarray, patch_size: int = 20) -> float:
    height, width = plane.shape
    side_y = min(patch_size, height)
    side_x = min(patch_size, width)
    patches = (
        plane[:side_y, :side_x],
        plane[:side_y, width - side_x:],
        plane[height - side_y:, :side_x],
        plane[height - side_y:, width - side_x:],
    )
    return float(np.mean([float(patch.mean()) for patch in patches]))


def _remove_small_components(mask: np.ndarray, minimum_size: int = 301) -> np.ndarray:
    # Four-neighbour connectivity matches the usual 2-D connectivity=1 rule.
    cross = np.asarray([[False, True, False], [True, True, True], [False, True, False]])
    labels, component_count = ndimage.label(mask, structure=cross)
    if component_count == 0:
        return mask.astype(bool, copy=False)
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def _segment_plane(plane: np.ndarray, *, sigma: float, closing_radius: int) -> np.ndarray:
    source = np.asarray(plane, dtype=np.float32)
    blurred = ndimage.gaussian_filter(source, sigma=sigma)
    mask = blurred > _otsu_threshold(blurred)

    # Orient the threshold using the combined mean of the four 20x20 corners.
    if _corner_mean(source) > float(source.mean()):
        mask = ~mask

    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask, structure=_disk(closing_radius), border_value=0)
    mask = _remove_small_components(mask, minimum_size=301)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, structure=np.ones((2, 2), dtype=bool), border_value=0)
    mask = ndimage.binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def otsu_grey_mask(rgb: np.ndarray) -> np.ndarray:
    """Gray intensity pipeline: Gaussian sigma 2, disk closing radius 8."""
    return _segment_plane(rgb_to_grey(rgb), sigma=2.0, closing_radius=8)


def otsu_saturation_mask(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation pipeline: Gaussian sigma 5, disk closing radius 9."""
    return _segment_plane(rgb_to_saturation(rgb), sigma=5.0, closing_radius=9)


def corner_foreground_fractions(mask: np.ndarray, patch_size: int = 20) -> tuple[float, ...]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    side_y = min(patch_size, height)
    side_x = min(patch_size, width)
    patches = (
        binary[:side_y, :side_x],
        binary[:side_y, width - side_x:],
        binary[height - side_y:, :side_x],
        binary[height - side_y:, width - side_x:],
    )
    return tuple(float(patch.mean()) for patch in patches)


def select_labelfree_variant(grey_mask: np.ndarray, saturation_mask: np.ndarray) -> str:
    """Choose the candidate with the clearest worst corner, then mean corners.

    Ties and invalid comparisons fall back to saturation, making selection
    deterministic and usable at inference time.
    """
    grey = corner_foreground_fractions(grey_mask)
    saturation = corner_foreground_fractions(saturation_mask)
    if len(grey) != 4 or len(saturation) != 4:
        return "saturation"
    if not np.isfinite(grey).all() or not np.isfinite(saturation).all():
        return "saturation"
    grey_score = (max(grey), float(np.mean(grey)))
    saturation_score = (max(saturation), float(np.mean(saturation)))
    return "grey" if grey_score < saturation_score else "saturation"


def make_mask(
    rgb: np.ndarray,
    mode: str,
    *,
    class_name: str | None = None,
) -> tuple[np.ndarray, str]:
    """Create a binary mask and report which Otsu branch produced it."""
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown G2 mask mode {mode!r}; choose from {', '.join(MASK_MODES)}.")
    if mode == "grey":
        return otsu_grey_mask(rgb), "grey"
    if mode == "saturation":
        return otsu_saturation_mask(rgb), "saturation"
    if mode == "published":
        if not class_name:
            raise ValueError("The published per-class mask mode requires a class name.")
        variant = "grey" if class_name in GREY_CLASSES else "saturation"
        return make_mask(rgb, variant, class_name=class_name)
    grey_mask = otsu_grey_mask(rgb)
    saturation_mask = otsu_saturation_mask(rgb)
    variant = select_labelfree_variant(grey_mask, saturation_mask)
    return (grey_mask if variant == "grey" else saturation_mask), variant


def mask_path(project: Path, row: dict[str, str], mode: str) -> Path:
    source = Path(row["path"].replace("\\", "/"))
    if source.suffix == "":
        raise ValueError(f"Image path has no filename extension: {row['path']}")
    return (
        project / "masks" / f"g2_{mode}" / row["split"] / row["class"]
        / f"{source.stem}.png"
    )


def _mask_metadata(image_path: Path, relative_path: str, mode: str, class_name: str) -> dict[str, object]:
    stat = image_path.stat()
    return {
        "schema": SEGMENTATION_SCHEMA,
        "source": relative_path.replace("\\", "/"),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "mode": mode,
        "class": class_name if mode == "published" else "",
    }


def get_or_create_mask(
    project: Path,
    row: dict[str, str],
    rgb: np.ndarray,
    mode: str,
    *,
    refresh: bool = False,
) -> tuple[np.ndarray, str]:
    """Read a current cached mask or create it atomically from the source image."""
    image_path = project / Path(row["path"].replace("\\", "/"))
    target = mask_path(project, row, mode)
    expected_meta = _mask_metadata(image_path, row["path"], mode, row["class"])
    if target.is_file() and not refresh:
        try:
            with Image.open(target) as stored:
                metadata = json.loads(stored.info.get("g2_metadata", "{}"))
                mask = np.asarray(stored.convert("L"), dtype=np.uint8) >= 128
            if (
                all(metadata.get(key) == value for key, value in expected_meta.items())
                and mask.shape == rgb.shape[:2]
            ):
                return mask, str(metadata.get("variant", mode))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    mask, variant = make_mask(rgb, mode, class_name=row["class"])
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**expected_meta, "variant": variant}
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("g2_metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    temporary = target.with_name(target.name + ".tmp")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        temporary, format="PNG", pnginfo=png_info
    )
    os.replace(temporary, target)
    return mask, variant


def _resolve(project: Path, path: Path) -> Path:
    return path if path.is_absolute() else project / path


def main(argv: list[str] | None = None) -> int:
    """Generate and cache G2 masks for the canonical split."""
    project = project_root()
    config = load_config(project / "configs" / "default.yaml")
    data_config = config["data"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--split-file", type=Path, default=Path(data_config["split_file"]))
    parser.add_argument("--classes-file", type=Path, default=Path(data_config["classes_file"]))
    parser.add_argument(
        "--modes", nargs="+", choices=MASK_MODES, default=["saturation", "grey", "labelfree"]
    )
    parser.add_argument("--refresh", action="store_true", help="Recompute cached masks.")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    split_rows, _ = read_split_contract(_resolve(root, args.split_file))
    class_order = read_lines(_resolve(root, args.classes_file))
    if len(class_order) != 68 or len(set(class_order)) != 68:
        raise ValueError("The G2 anchor requires the fixed 68-class order.")
    for mode in args.modes:
        for row in split_rows:
            source = root / Path(row["path"].replace("\\", "/"))
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            get_or_create_mask(root, row, rgb, mode, refresh=args.refresh)
        print(f"Cached {len(split_rows)} {mode} masks under masks/g2_{mode}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

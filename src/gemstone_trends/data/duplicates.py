"""Audit near-duplicate image candidates using pHash and a visual descriptor."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

from gemstone_trends.config import load_config
from gemstone_trends.data.splits import SPLIT_COLUMNS, project_root


PAIR_COLUMNS = (
    "path_a", "class_a", "split_a", "path_b", "class_b", "split_b",
    "phash_distance", "embedding_cosine", "phash_flag", "embedding_flag",
    "high_confidence", "requires_review",
)


def _image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.copy()
        # Some Kaggle files have malformed EXIF metadata which Pillow's
        # exif_transpose tries to serialize while rotating. Read the orientation
        # tag only, then apply the pixel transform without rewriting EXIF.
        orientation = handle.getexif().get(274)
    operations = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    if orientation in operations:
        image = image.transpose(operations[orientation])
    return image.convert("RGB")


def _phash(image: Image.Image) -> np.uint64:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(gray, type=2, norm="ortho")[:8, :8]
    values = low.reshape(-1)[1:]
    median = float(np.median(values))
    result = 0
    for index, value in enumerate(values):
        if value > median:
            result |= 1 << index
    return np.uint64(result)


def _embedding(image: Image.Image) -> np.ndarray:
    """Return a compact HOG + RGB histogram embedding (not a trained CNN feature)."""
    rgb = np.asarray(image.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.minimum((orientation * (9.0 / np.pi)).astype(np.int32), 8)

    # 8x8 spatial cells with nine unsigned gradient-orientation bins each.
    hog = np.zeros((8, 8, 9), dtype=np.float32)
    cell_y = np.arange(64)[:, None] // 8
    cell_x = np.arange(64)[None, :] // 8
    for orientation_bin in range(9):
        selected = bins == orientation_bin
        np.add.at(
            hog[:, :, orientation_bin],
            (np.broadcast_to(cell_y, (64, 64))[selected], np.broadcast_to(cell_x, (64, 64))[selected]),
            magnitude[selected],
        )
    hog = hog.reshape(-1)
    hog /= max(float(np.linalg.norm(hog)), 1e-12)

    color_parts = []
    for channel in range(3):
        histogram, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0))
        color_parts.extend(histogram.astype(np.float32) / rgb[:, :, channel].size)
    color = np.sqrt(np.asarray(color_parts, dtype=np.float32))
    color /= max(float(np.linalg.norm(color)), 1e-12)

    vector = np.concatenate((0.7 * hog, 0.3 * color)).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector


def _read_split(split_file: Path) -> list[dict[str, str]]:
    with split_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
            raise ValueError(f"{split_file} must have columns: {','.join(SPLIT_COLUMNS)}")
        return list(reader)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _review_sheet(
    rows: list[dict[str, object]], root: Path, output: Path, limit: int
) -> None:
    font_size = 12
    tile_w, tile_h, caption_h = 220, 165, 45
    chosen = sorted(
        rows,
        key=lambda row: (
            str(row["high_confidence"]).lower() != "true",
            int(row["phash_distance"]),
            -float(row["embedding_cosine"]),
        ),
    )[:limit]
    canvas = Image.new("RGB", (tile_w * 2, max(1, len(chosen)) * (tile_h + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, pair in enumerate(chosen):
        top = index * (tile_h + caption_h)
        for side, key in enumerate(("path_a", "path_b")):
            image = _image(root / str(pair[key]))
            image.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
            x = side * tile_w + (tile_w - image.width) // 2
            y = top + (tile_h - image.height) // 2
            canvas.paste(image, (x, y))
        caption = (
            f"{pair['class_a']} ({pair['split_a']}) ↔ {pair['class_b']} ({pair['split_b']})\n"
            f"pHash={pair['phash_distance']} cosine={float(pair['embedding_cosine']):.4f}"
        )
        draw.text((5, top + tile_h + 2), caption, fill="black", font_size=font_size)
    if not chosen:
        draw.text((10, 10), "No duplicate candidates at the configured thresholds.", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def run_audit(
    *,
    root: Path,
    split_file: Path,
    output_dir: Path,
    phash_max_distance: int = 4,
    embedding_min_cosine: float = 0.99,
    review_sample_size: int = 40,
    folds: int = 5,
    fold_seed: int = 42,
    high_confidence_requires_both: bool = True,
    update_split: bool = True,
) -> dict[str, int]:
    rows = _read_split(split_file)
    if not rows:
        raise ValueError("The split file is empty.")
    paths = [row["path"] for row in rows]
    hashes: list[np.uint64] = []
    embeddings: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        image = _image(root / row["path"])
        hashes.append(_phash(image))
        embeddings.append(_embedding(image))
        if index % 500 == 0 or index == len(rows):
            print(f"Fingerprinted {index}/{len(rows)} images")

    hash_array = np.asarray(hashes, dtype=np.uint64)
    xor = np.bitwise_xor(hash_array[:, None], hash_array[None, :])
    byte_view = xor.view(np.uint8).reshape(len(rows), len(rows), 8)
    popcount = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    distances = popcount[byte_view].sum(axis=2, dtype=np.uint16)
    vectors = np.stack(embeddings)
    similarities = vectors @ vectors.T
    np.clip(similarities, -1.0, 1.0, out=similarities)

    uf = _UnionFind(len(rows))
    candidate_pairs: list[dict[str, object]] = []
    cross_split_count = 0
    high_confidence_count = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            phash_distance = int(distances[left, right])
            cosine = float(similarities[left, right])
            phash_flag = phash_distance <= phash_max_distance
            embedding_flag = cosine >= embedding_min_cosine
            if not (phash_flag or embedding_flag):
                continue
            high_confidence = (
                phash_flag and embedding_flag
                if high_confidence_requires_both
                else phash_flag or embedding_flag
            )
            if high_confidence:
                uf.union(left, right)
                high_confidence_count += 1
            if rows[left]["split"] != rows[right]["split"]:
                cross_split_count += 1
            candidate_pairs.append({
                "path_a": paths[left], "class_a": rows[left]["class"],
                "split_a": rows[left]["split"], "path_b": paths[right],
                "class_b": rows[right]["class"], "split_b": rows[right]["split"],
                "phash_distance": phash_distance,
                "embedding_cosine": f"{cosine:.8f}",
                "phash_flag": str(phash_flag).lower(),
                "embedding_flag": str(embedding_flag).lower(),
                "high_confidence": str(high_confidence).lower(),
                "requires_review": "true",
            })

    candidate_pairs.sort(
        key=lambda row: (
            row["high_confidence"] != "true",
            int(row["phash_distance"]),
            -float(row["embedding_cosine"]),
            str(row["path_a"]),
            str(row["path_b"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "near_duplicate_candidates.csv"
    cross_path = output_dir / "train_test_near_duplicate_candidates.csv"
    _write_csv(all_path, PAIR_COLUMNS, candidate_pairs)
    cross_rows = [row for row in candidate_pairs if row["split_a"] != row["split_b"]]
    _write_csv(cross_path, PAIR_COLUMNS, cross_rows)
    _review_sheet(
        candidate_pairs, root, output_dir / "near_duplicate_review.jpg", review_sample_size
    )

    components: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        components.setdefault(uf.find(index), []).append(row["path"])
    ordered_components = sorted(components.values(), key=lambda values: min(values).casefold())
    group_by_path: dict[str, str] = {}
    for group_index, component in enumerate(ordered_components, start=1):
        group_name = f"dup-{group_index:05d}"
        for path in component:
            group_by_path[path] = group_name
    if update_split:
        for row in rows:
            row["dup_group"] = group_by_path[row["path"]]
        train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
        train_labels = np.asarray([rows[index]["class"] for index in train_indices], dtype=object)
        train_groups = np.asarray([rows[index]["dup_group"] for index in train_indices], dtype=object)
        fold_builder = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
        for fold_index, (_, validation_indices) in enumerate(
            fold_builder.split(np.zeros(len(train_indices)), train_labels, train_groups)
        ):
            for local_index in validation_indices:
                rows[train_indices[int(local_index)]]["fold"] = str(fold_index)
        _write_csv(split_file, SPLIT_COLUMNS, rows)

    members_by_group: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        members_by_group.setdefault(row["dup_group"], []).append(row)
    exclusions = []
    for group_name, members in sorted(members_by_group.items()):
        train_members = [member for member in members if member["split"] == "train"]
        test_members = [member for member in members if member["split"] == "test"]
        if train_members and test_members:
            train_paths = ";".join(member["path"] for member in train_members)
            for test_member in test_members:
                exclusions.append({
                    "path": test_member["path"],
                    "dup_group": group_name,
                    "train_paths": train_paths,
                })
    _write_csv(
        output_dir / "test_duplicates_to_exclude.csv",
        ("path", "dup_group", "train_paths"),
        exclusions,
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(rows),
        "candidate_pairs": len(candidate_pairs),
        "train_test_candidate_pairs": len(cross_rows),
        "high_confidence_pairs": high_confidence_count,
        "duplicate_groups": len(ordered_components),
        "duplicate_groups_with_multiple_images": sum(len(group) > 1 for group in ordered_components),
        "train_test_images_to_exclude": len(exclusions),
        "phash_max_distance": phash_max_distance,
        "embedding_min_cosine": embedding_min_cosine,
        "folds": folds,
        "fold_seed": fold_seed,
        "embedding_descriptor": "64x64 unsigned-gradient HOG plus RGB histogram; cosine similarity",
        "high_confidence_requires_both": high_confidence_requires_both,
        "split_updated_with_dup_group": update_split,
        "train_folds_grouped_after_audit": update_split,
        "thresholds_reviewed_by_human": False,
    }
    summary_path = output_dir / "near_duplicate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "images": len(rows),
        "candidate_pairs": len(candidate_pairs),
        "train_test_candidate_pairs": len(cross_rows),
        "high_confidence_pairs": high_confidence_count,
        "duplicate_groups_with_multiple_images": summary["duplicate_groups_with_multiple_images"],
    }


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    config = load_config()
    data_config = config["data"]
    audit_config = config["duplicate_audit"]
    project_config = config["project"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phash-max-distance", type=int)
    parser.add_argument("--embedding-min-cosine", type=float)
    parser.add_argument("--review-sample-size", type=int)
    parser.add_argument("--no-update-split", action="store_true")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project / path
    split_file = resolve(args.split_file or Path(data_config["split_file"]))
    output_dir = resolve(args.output_dir or Path("results/audits"))
    summary = run_audit(
        root=project,
        split_file=split_file,
        output_dir=output_dir,
        phash_max_distance=(args.phash_max_distance if args.phash_max_distance is not None else int(audit_config["phash_max_distance"])),
        embedding_min_cosine=(args.embedding_min_cosine if args.embedding_min_cosine is not None else float(audit_config["embedding_min_cosine"])),
        review_sample_size=(args.review_sample_size if args.review_sample_size is not None else int(audit_config["review_sample_size"])),
        folds=int(data_config["folds"]),
        fold_seed=int(project_config["seed"]),
        high_confidence_requires_both=bool(audit_config["high_confidence_requires_both"]),
        update_split=not args.no_update_split,
    )
    print(json.dumps(summary, indent=2))
    print("Inspect results/audits/near_duplicate_review.jpg; thresholds are not yet reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

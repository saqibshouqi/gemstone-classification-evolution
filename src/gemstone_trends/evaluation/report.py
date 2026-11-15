"""Score prediction files and produce the shared coursework scoreboard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from gemstone_trends.config import load_config
from gemstone_trends.data.splits import project_root, read_lines
from gemstone_trends.evaluation.io import (
    _prediction_files,
    read_split_contract,
    validate_predictions,
)
from gemstone_trends.evaluation.metrics import (
    bootstrap_accuracy_interval,
    classification_metrics,
    exact_mcnemar,
    risk_coverage_rows,
)


FILENAME_PATTERN = re.compile(r"^(g[123])_(.+)_([^_]+)_s(\d+)$")
METRIC_NAMES = (
    "top1_accuracy", "top5_accuracy", "macro_f1", "ece", "multiclass_brier",
    "bootstrap_95_low", "bootstrap_95_high",
)
MCNEMAR_COLUMNS = (
    "where", "mode", "arm_a", "model_a", "seed_a", "arm_b", "model_b", "seed_b",
    "a_correct_b_wrong", "a_wrong_b_correct", "discordant", "mcnemar_exact_p",
    "source_a", "source_b",
)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    fieldnames = list(columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_name(path: Path) -> tuple[str, str, str, int]:
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if not match:
        raise ValueError(
            f"Prediction filename {path.name!r} must match "
            "<g1|g2|g3>_<model>_<mode>_s<seed>.csv."
        )
    arm, model, mode, seed = match.groups()
    return arm, model, mode, int(seed)


def _micro_metadata(rows: list[dict[str, object]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    paths = [str(row["path"]) for row in rows]
    labels = [str(row["true"]) for row in rows]
    return paths, np.asarray(labels, dtype=object), np.stack(
        [np.asarray(row["probabilities"], dtype=np.float64) for row in rows]
    )


def _confusion_rows(confusion: np.ndarray, classes: list[str]) -> list[dict[str, object]]:
    return [
        {"true_class": classes[row], **{classes[col]: int(confusion[row, col]) for col in range(len(classes))}}
        for row in range(len(classes))
    ]


def _per_class_rows(confusion: np.ndarray, classes: list[str], metadata: dict[str, object]) -> list[dict[str, object]]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    output = []
    for index, class_name in enumerate(classes):
        recall = float(true_positive[index] / support[index]) if support[index] else 0.0
        precision = float(true_positive[index] / predicted[index]) if predicted[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.append({
            **metadata,
            "class": class_name,
            "support": int(support[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return output


def _generation_chart(path: Path, mode: str, arm_rows: list[dict[str, object]]) -> None:
    colors = {"g1": "#6b7280", "g2": "#2563eb", "g3": "#16a34a"}
    x_positions = {"g1": 110, "g2": 300, "g3": 490}
    width, height = 600, 420
    left, top, plot_width, plot_height = 70, 35, 480, 320
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in arm_rows:
        grouped[str(row["arm"])].append(float(row["top1_mean"]))
    present = [arm for arm in ("g1", "g2", "g3") if grouped.get(arm)]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">Top-1 accuracy by generation ({mode}, test)</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_height * (1 - tick / 100)
        chunks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        chunks.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}%</text>')
    chunks.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#374151"/>')
    chunks.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#374151"/>')
    points = []
    for arm in present:
        mean = float(np.mean(grouped[arm]))
        x = x_positions[arm]
        y = top + plot_height * (1 - mean)
        points.append(f"{x},{y:.1f}")
        chunks.append(f'<circle cx="{x}" cy="{y:.1f}" r="7" fill="{colors[arm]}"/>')
        chunks.append(f'<text x="{x}" y="{top+plot_height+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{arm.upper()} · {mean:.1%}</text>')
        if len(grouped[arm]) > 1:
            spread = statistics.stdev(grouped[arm])
            y1 = top + plot_height * (1 - min(1.0, mean + spread))
            y2 = top + plot_height * (1 - max(0.0, mean - spread))
            chunks.append(f'<line x1="{x}" y1="{y1:.1f}" x2="{x}" y2="{y2:.1f}" stroke="{colors[arm]}" stroke-width="2"/>')
    if len(points) > 1:
        chunks.insert(3, f'<polyline points="{" ".join(points)}" fill="none" stroke="#9ca3af" stroke-width="2"/>')
    chunks.append('<text x="300" y="402" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4b5563">Point = average across model variants; bars show between-variant SD when available.</text>')
    chunks.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def evaluate_files(
    *,
    files: list[Path],
    root: Path,
    split_file: Path,
    classes_file: Path,
    table_dir: Path,
    figure_dir: Path,
    bootstrap_resamples: int = 1000,
    ece_bins: int = 15,
    bootstrap_seed: int = 0,
    allow_partial: bool = False,
) -> dict[str, int]:
    classes = read_lines(classes_file)
    if len(classes) != 68 or len(set(classes)) != len(classes):
        raise ValueError(f"Expected 68 unique classes in {classes_file}.")
    class_index = {name: index for index, name in enumerate(classes)}
    split_rows, split_by_path = read_split_contract(split_file)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split_row in split_rows:
        group_splits[split_row["dup_group"]].add(split_row["split"])
    duplicate_test_paths = {
        path
        for path, split_row in split_by_path.items()
        if split_row["split"] == "test"
        and group_splits[split_row["dup_group"]] >= {"train", "test"}
    }

    metric_rows: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []

    for prediction_file in files:
        arm, model, mode, seed = _parse_name(prediction_file)
        predictions = validate_predictions(
            prediction_file,
            split_rows,
            split_by_path,
            classes,
            allow_partial=allow_partial,
        )
        test_rows = [row for row in predictions if row["where"] == "test"]
        oof_rows = [row for row in predictions if row["where"] == "oof"]
        score_groups: list[tuple[str, list[dict[str, object]], int]] = []
        if test_rows:
            score_groups.append(("test", test_rows, 0))
            kept_test_rows = [row for row in test_rows if str(row["path"]) not in duplicate_test_paths]
            if kept_test_rows:
                score_groups.append(("test_no_dup", kept_test_rows, len(test_rows) - len(kept_test_rows)))
        if oof_rows:
            score_groups.append(("oof", oof_rows, 0))

        for where, rows, duplicates_removed in score_groups:
            paths, labels, probabilities = _micro_metadata(rows)
            true_indices = np.asarray([class_index[str(label)] for label in labels], dtype=np.int64)
            metrics, confusion, _recall = classification_metrics(
                true_indices, probabilities, class_count=len(classes), ece_bins=ece_bins
            )
            suffix = int(hashlib.sha256(f"{prediction_file.name}:{where}".encode()).hexdigest()[:8], 16)
            ci_low, ci_high = bootstrap_accuracy_interval(
                true_indices,
                probabilities,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + suffix,
            )
            metadata: dict[str, object] = {
                "arm": arm, "model": model, "mode": mode, "seed": seed,
                "where": where, "source_file": prediction_file.name,
                "duplicate_test_images_removed": duplicates_removed,
            }
            metric_rows.append({
                **metadata,
                "n": int(metrics["n"]),
                "top1_accuracy": metrics["top1_accuracy"],
                "top5_accuracy": metrics["top5_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "multiclass_brier": metrics["multiclass_brier"],
                "bootstrap_95_low": ci_low,
                "bootstrap_95_high": ci_high,
            })
            per_class.extend(_per_class_rows(confusion, classes, metadata))
            for entry in risk_coverage_rows(true_indices, probabilities):
                risk_rows.append({**metadata, **entry})
            if where in {"test", "test_no_dup"}:
                confusion_path = table_dir / "confusion" / f"{prediction_file.stem}_{where}.csv"
                _write_csv(confusion_path, _confusion_rows(confusion, classes))
            if where in {"test", "test_no_dup"}:
                evaluated.append({
                    **metadata,
                    "paths": paths,
                    "labels": labels,
                    "true_indices": true_indices,
                    "probabilities": probabilities,
                    "correct": np.argmax(probabilities, axis=1) == true_indices,
                })

    if not metric_rows:
        raise ValueError("No usable test or OOF rows were found in the prediction files.")

    aggregate_map: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        key = (str(row["arm"]), str(row["model"]), str(row["mode"]), str(row["where"]))
        aggregate_map[key].append(row)
    scoreboard: list[dict[str, object]] = []
    for key, group in sorted(aggregate_map.items()):
        aggregate: dict[str, object] = {
            "arm": key[0], "model": key[1], "mode": key[2], "where": key[3],
            "seed_count": len(group),
            "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda row: int(row["seed"]))),
            "duplicate_test_images_removed": int(group[0]["duplicate_test_images_removed"]),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group]
            aggregate[f"{metric}_mean"] = float(statistics.mean(values))
            aggregate[f"{metric}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        aggregate["n_images"] = int(group[0]["n"])
        scoreboard.append(aggregate)

    pairwise: list[dict[str, object]] = []
    for left_index, left in enumerate(evaluated):
        for right in evaluated[left_index + 1 :]:
            if left["where"] != right["where"] or left["mode"] != right["mode"]:
                continue
            if set(left["paths"]) != set(right["paths"]):
                continue
            right_positions = {path: index for index, path in enumerate(right["paths"])}
            aligned_right = np.asarray(
                [right_positions[path] for path in left["paths"]], dtype=np.int64
            )
            if not np.array_equal(left["labels"], right["labels"][aligned_right]):
                continue
            a_only, b_only, p_value = exact_mcnemar(
                left["correct"], right["correct"][aligned_right]
            )
            pairwise.append({
                "where": left["where"], "mode": left["mode"],
                "arm_a": left["arm"], "model_a": left["model"], "seed_a": left["seed"],
                "arm_b": right["arm"], "model_b": right["model"], "seed_b": right["seed"],
                "a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only,
                "discordant": a_only + b_only, "mcnemar_exact_p": p_value,
                "source_a": left["source_file"], "source_b": right["source_file"],
            })

    _write_csv(table_dir / "per_seed_metrics.csv", metric_rows)
    _write_csv(table_dir / "scoreboard.csv", scoreboard)
    _write_csv(table_dir / "per_class_metrics.csv", per_class)
    _write_csv(table_dir / "risk_coverage.csv", risk_rows)
    _write_csv(table_dir / "mcnemar_exact.csv", pairwise, columns=MCNEMAR_COLUMNS)

    for where in ("test", "test_no_dup"):
        for mode in sorted({str(row["mode"]) for row in scoreboard if row["where"] == where}):
            mode_rows = [row for row in scoreboard if row["where"] == where and row["mode"] == mode]
            _generation_chart(figure_dir / f"trend_{mode}_{where}.svg", mode, mode_rows)

    return {
        "prediction_files": len(files),
        "scored_groups": len(metric_rows),
        "scoreboard_rows": len(scoreboard),
        "paired_mcnemar_tests": len(pairwise),
    }


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
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--ece-bins", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else project / path
    files = _prediction_files(
        [resolve(path) for path in (args.paths or [Path(evaluation_config["prediction_directory"])])]
    )
    table_dir = resolve(args.table_dir or Path(evaluation_config["table_directory"]))
    figure_dir = resolve(args.figure_dir or Path(evaluation_config["figure_directory"]))
    summary = evaluate_files(
        files=files,
        root=project,
        split_file=resolve(args.split_file or Path(data_config["split_file"])),
        classes_file=resolve(args.classes_file or Path(data_config["classes_file"])),
        table_dir=table_dir,
        figure_dir=figure_dir,
        bootstrap_resamples=(
            args.bootstrap_resamples
            if args.bootstrap_resamples is not None
            else int(evaluation_config["bootstrap_resamples"])
        ),
        ece_bins=args.ece_bins if args.ece_bins is not None else int(evaluation_config["ece_bins"]),
        bootstrap_seed=args.bootstrap_seed,
        allow_partial=args.allow_partial,
    )
    print("Scoreboard updated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  table: {(table_dir / 'scoreboard.csv').relative_to(project).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

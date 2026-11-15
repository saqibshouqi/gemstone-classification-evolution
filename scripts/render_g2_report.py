"""Render the G2 reproduction report from the shared scoreboard CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD = ROOT / "results" / "tables" / "scoreboard.csv"
OUTPUT = ROOT / "docs" / "g2_reproduction_report.md"

CLASSIFIER_ORDER = ("rf", "lr", "svm", "nb", "lda", "knn", "dt")
CLASSIFIER_LABELS = {
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "svm": "One-vs-rest SVM",
    "nb": "Gaussian Naive Bayes",
    "lda": "Linear Discriminant Analysis",
    "knn": "k-Nearest Neighbours",
    "dt": "Decision Tree",
}
FEATURE_ORDER = ("rf", "rf_rgb8", "rf_lbp8_r1")
FEATURE_LABELS = {
    "rf": "RGB histogram + LBP",
    "rf_rgb8": "RGB histogram only",
    "rf_lbp8_r1": "LBP only",
}
MODE_LABELS = {"labelfree": "Label-free", "published": "Published oracle"}
PAPER_REFERENCE = {
    "rf": (69.4, 94.4),
    "lr": (68.7, 92.6),
    "svm": (66.9, 86.3),
    "nb": (62.7, 77.8),
    "lda": (59.9, 94.0),
    "knn": (54.6, 85.9),
    "dt": (46.5, 73.9),
}


def _metric(row: dict[str, str], metric: str) -> str:
    mean = float(row[f"{metric}_mean"]) * 100
    std = float(row[f"{metric}_std"]) * 100
    return f"{mean:.2f}% +/- {std:.2f}"


def _rows(
    data: list[dict[str, str]], *, where: str, models: tuple[str, ...]
) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["model"], row["mode"]): row
        for row in data
        if row["where"] == where and row["model"] in models
    }


def _table_header() -> list[str]:
    return [
        "| Model / feature view | Mask mode | N | Top-1 accuracy | Top-5 accuracy | Macro-F1 | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]


def _table_rows(
    lookup: dict[tuple[str, str], dict[str, str]],
    models: tuple[str, ...],
    modes: tuple[str, ...],
    labels: dict[str, str],
) -> list[str]:
    output: list[str] = []
    for model in models:
        for mode in modes:
            row = lookup.get((model, mode))
            if row is None:
                raise ValueError(f"Missing scoreboard row for {model}/{mode}.")
            output.append(
                f"| {labels[model]} | {MODE_LABELS[mode]} | {row['n_images']} | "
                f"{_metric(row, 'top1_accuracy')} | {_metric(row, 'top5_accuracy')} | "
                f"{_metric(row, 'macro_f1')} | {_metric(row, 'ece')} |"
            )
    return output


def render() -> str:
    if not SCOREBOARD.is_file():
        raise FileNotFoundError(f"Run scripts/evaluate.py first; missing {SCOREBOARD}.")
    with SCOREBOARD.open("r", encoding="utf-8-sig", newline="") as handle:
        data = list(csv.DictReader(handle))

    full = _rows(data, where="test", models=(*CLASSIFIER_ORDER, *FEATURE_ORDER[1:]))
    deduplicated = _rows(
        data, where="test_no_dup", models=(*CLASSIFIER_ORDER, *FEATURE_ORDER[1:])
    )
    all_modes = ("labelfree", "published")

    lines = [
        "# G2: 2022 gemstone classification reproduction",
        "",
        "## Scope and protocol",
        "",
        "This report covers the core G2 package in the implementation design: both documented Otsu branches, the label-free mask selector, the seven classical classifiers, and E3's histogram-only / LBP-only / combined-feature ablation. The comparison uses the fixed 68-class Chow split (2,042 train and 284 test images), five predefined CV folds for parameter selection, SMOTE with k=4 inside the training pipeline, and five stochastic seeds for test-score aggregation.",
        "",
        "The combined feature vector has 512 normalized joint RGB histogram values and 256 LBP values. The E3 views select the first 512 values, the last 256 values, or all 768. Grid selection uses negative log loss on the fixed five-fold split; the held-out test set is used only for final scoring.",
        "",
        "**Mask modes:** `labelfree` selects between gray and saturation Otsu candidates using corner foreground occupancy and is the leakage-free result. `published` selects the branch using each image's known class label, so its scores are oracle diagnostics and must not be treated as deployable performance. The two modes are reported in separate rows throughout.",
        "",
        "The supplied dataset contains raw source images, not the authors' saved `*_masked` files. Masks are reconstructed from the documented Otsu recipes, so this is a recipe-level reproduction rather than a pixel-exact reproduction of the original masks.",
        "",
        "The SVM's full four-kernel / three-C / two-gamma grid was searched with five-fold CV once per mask mode using seed 0. Seeds 1–4 refit SMOTE and the classifier using that mode's saved CV-selected parameters. Their summaries record `fixed_params_from_seed_grid` and `tuning_source_seed: 0`; this avoids repeating the unusually expensive probability-calibrated OVR grid while retaining five stochastic test predictions per mode. The other model and feature-view runs performed their CV selection for each seed.",
        "",
        "## Classifier comparison: full test set",
        "",
        "Values are mean +/- standard deviation over seeds 0–4. Accuracy, macro-F1, and ECE are percentages; lower ECE is better.",
        "",
        *_table_header(),
    ]
    lines.extend(_table_rows(full, CLASSIFIER_ORDER, all_modes, CLASSIFIER_LABELS))

    lines.extend([
        "",
        "## E3 feature ablation: full test set",
        "",
        *_table_header(),
    ])
    lines.extend(_table_rows(full, FEATURE_ORDER, all_modes, FEATURE_LABELS))

    lines.extend([
        "",
        "## Duplicate-excluded test set",
        "",
        "The evaluator excludes the two test images marked by the frozen high-confidence duplicate audit, leaving 282 images. The full-test rows above remain the primary scores.",
        "",
        *_table_header(),
    ])
    lines.extend(_table_rows(deduplicated, CLASSIFIER_ORDER, all_modes, CLASSIFIER_LABELS))
    lines.extend(_table_rows(deduplicated, FEATURE_ORDER[1:], all_modes, FEATURE_LABELS))

    lines.extend([
        "",
        "## Comparison with the paper reference",
        "",
        "The design document records the paper's reported top-1 / top-5 reference values. The table compares them to this reproduction's leakage-free label-free results; the paper's values and our reconstructed-mask results are not expected to match exactly. The design also notes that the paper selected the best result among 231 candidates on its test set, which can make those reference values optimistic. This reproduction selected parameters only by the fixed training folds and never by held-out test scores.",
        "",
        "| Classifier | Paper top-1 / top-5 | Reproduced label-free top-1 | Reproduced label-free top-5 |",
        "|---|---:|---:|---:|",
    ])
    for model in CLASSIFIER_ORDER:
        row = full[(model, "labelfree")]
        paper1, paper5 = PAPER_REFERENCE[model]
        lines.append(
            f"| {CLASSIFIER_LABELS[model]} | {paper1:.1f}% / {paper5:.1f}% | "
            f"{_metric(row, 'top1_accuracy')} | {_metric(row, 'top5_accuracy')} |"
        )

    lines.extend([
        "",
        "## Main observations",
        "",
        "- Among leakage-free combined-feature classifiers, Random Forest has the highest mean top-1 accuracy (65.56%) and top-5 accuracy (93.94%).",
        "- In the RF E3 ablation, RGB histogram only reaches 64.58% top-1, while LBP only reaches 21.34%; the combined view reaches 65.56%. This dataset result shows most of the RF discrimination comes from the color histogram, with the combined descriptor adding about 0.98 percentage points over histogram only.",
        "- Published-oracle RF reaches 68.17% top-1. The gain over label-free RF is an expected diagnostic consequence of using the test image's class to choose its mask; it is not a fair operational performance estimate.",
        "- The label-free RF score changes from 65.56% on all 284 test images to 65.32% on the 282-image duplicate-excluded subset.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `results/tables/scoreboard.csv`: all 36 aggregate rows, including all metrics for full and duplicate-excluded test sets.",
        "- `results/tables/per_seed_metrics.csv`: per-seed scores for every prediction file.",
        "- `results/tables/g2/<mode>/`: ranked CV grid tables and per-seed summaries; SVM seed 1–4 summaries link to their mode's seed 0 grid.",
        "- `results/preds/`: 90 validated, shared-contract prediction files.",
        "- `results/figures/trend_<mode>_<test-set>.svg`: four G2 comparison charts.",
        "- `results/tables/mcnemar_exact.csv`, `per_class_metrics.csv`, `risk_coverage.csv`, and `confusion/`: supporting evaluation outputs.",
        "- `models/g2/<mode>/`: local refittable model artifacts, excluded from Git.",
        "",
        "To regenerate the aggregates, run `python scripts/evaluate.py`; to regenerate this report from the scoreboard, run `python scripts/render_g2_report.py`.",
        "",
        "## Scope boundary",
        "",
        "The 33-feature-method grid is marked stretch work in the design. This reproduction completes the core E3 three-way histogram/LBP ablation, not the stretch grid or the other 2015/2025 method packages.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    report = render()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8", newline="\n")
    print(f"G2 report written to {OUTPUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

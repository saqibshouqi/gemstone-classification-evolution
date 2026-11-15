"""Train and score the G2 anchor on the fixed Chow 68-class benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from gemstone_trends.config import load_config
from gemstone_trends.paths import project_root
from g2.features import (
    FEATURE_LENGTH,
    FEATURE_NAME,
    FEATURE_VIEWS,
    feature_view_length,
    load_feature_dataset,
    select_feature_view,
)
from g2.models import MODEL_NAMES, aligned_probabilities, make_pipeline, require_imblearn
from g2.segmentation import MASK_MODES


def _write_csv(path: Path, rows: list[dict[str, object]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_predictions(
    path: Path,
    *,
    test_paths: np.ndarray,
    test_labels: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = ("path", "true", "where", *classes)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for image_path, label, row_probabilities in zip(test_paths, test_labels, probabilities):
            row: dict[str, object] = {
                "path": str(image_path),
                "true": str(label),
                "where": "test",
            }
            row.update({
                name: format(float(row_probabilities[index]), ".12g")
                for index, name in enumerate(classes)
            })
            writer.writerow(row)
    os.replace(temporary, path)


def train_one(
    *,
    root: Path,
    model: str,
    mode: str,
    seed: int,
    jobs: int,
    refresh_features: bool,
    feature_name: str = FEATURE_NAME,
    allow_published_oracle: bool = False,
    reuse_params_from_seed: int | None = None,
) -> dict[str, object]:
    if mode == "published" and not allow_published_oracle:
        raise ValueError(
            "The published per-class segmentation choice uses the known class label. "
            "It is diagnostic-only. Pass allow_published_oracle=True only for the explicitly "
            "label-conditioned reproduction comparison."
        )
    feature_length = feature_view_length(feature_name)
    config = load_config(root / "configs" / "default.yaml")
    g2_config = config.get("g2", {})
    dataset = load_feature_dataset(project=root, mode=mode, refresh=refresh_features)
    train_mask = dataset.splits == "train"
    test_mask = dataset.splits == "test"
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("The canonical split must contain both train and test images.")
    feature_matrix = select_feature_view(dataset.x, feature_name)
    train_x = feature_matrix[train_mask]
    test_x = feature_matrix[test_mask]
    class_index = {name: index for index, name in enumerate(dataset.classes)}
    try:
        train_y = np.asarray(
            [class_index[name] for name in dataset.labels[train_mask]], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(f"Unexpected label in canonical split: {exc.args[0]}") from exc
    test_y = dataset.labels[test_mask]
    test_paths = dataset.paths[test_mask]
    fold_ids = dataset.folds[train_mask]
    expected_folds = int(g2_config.get("cv_folds", 5))
    observed_folds = sorted(int(value) for value in np.unique(fold_ids))
    if observed_folds != list(range(expected_folds)):
        raise ValueError(
            f"Expected fixed training folds 0..{expected_folds - 1}; found {observed_folds}."
        )
    if np.any(fold_ids < 0):
        raise ValueError("Training rows must have a fold; test rows must not be used in CV.")
    cv = PredefinedSplit(test_fold=fold_ids)
    neighbours = int(g2_config.get("smote_neighbors", 4))
    pipeline, parameter_grid = make_pipeline(
        model, seed=seed, smote_neighbors=neighbours
    )
    scoring = str(g2_config.get("scoring", "neg_log_loss"))
    print(
        f"Searching {model.upper()} on {mode}: {len(train_x)} training rows, "
        f"{len(test_x)} held-out rows, {expected_folds} predefined folds, "
        f"{feature_length} {feature_name} features."
    )
    experiment_name = model if feature_name == FEATURE_NAME else f"{model}_{feature_name}"
    if reuse_params_from_seed is None:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=jobs,
            return_train_score=True,
            error_score="raise",
            verbose=1 if jobs == 1 else 0,
        )
        search.fit(train_x, train_y)
        fitted_pipeline = search.best_estimator_
        best_params = search.best_params_
        best_cv_log_loss = float(-search.best_score_)
        tuning_source_seed = seed
        results = search.cv_results_
        search_rows = []
        for index, parameters in enumerate(results["params"]):
            search_rows.append({
                "rank": int(results["rank_test_score"][index]),
                "mean_log_loss": float(-results["mean_test_score"][index]),
                "std_log_loss": float(results["std_test_score"][index]),
                "mean_fit_time_seconds": float(results["mean_fit_time"][index]),
                "params": json.dumps(parameters, sort_keys=True, default=str),
            })
        search_rows.sort(key=lambda row: (int(row["rank"]), float(row["mean_log_loss"])))
        search_path = (
            root / "results" / "tables" / "g2" / mode / f"{experiment_name}_s{seed}_grid.csv"
        )
        _write_csv(
            search_path,
            search_rows,
            ("rank", "mean_log_loss", "std_log_loss", "mean_fit_time_seconds", "params"),
        )
        selection_method = "five_fold_grid_search"
    else:
        source_model_path = (
            root / "models" / "g2" / mode / f"{experiment_name}_s{reuse_params_from_seed}.joblib"
        )
        if not source_model_path.is_file():
            raise FileNotFoundError(f"Cannot reuse missing tuning model: {source_model_path}")
        source_artifact = joblib.load(source_model_path)
        if not isinstance(source_artifact, dict):
            raise ValueError("The selected SVM tuning artifact is not a G2 artifact.")
        source_cv = source_artifact.get("cv", {}) if isinstance(source_artifact, dict) else {}
        if (
            source_artifact.get("model") != model
            or source_artifact.get("mode") != mode
            or source_artifact.get("feature_name") != feature_name
            or source_cv.get("feature_fingerprint") != dataset.fingerprint
        ):
            raise ValueError("The selected SVM tuning artifact does not match this data/mode/features.")
        best_params = dict(source_artifact["best_params"])
        pipeline.set_params(**best_params)
        fitted_pipeline = pipeline.fit(train_x, train_y)
        best_cv_log_loss = -float(source_cv["best_mean_neg_log_loss"])
        tuning_source_seed = reuse_params_from_seed
        search_path = (
            root / "results" / "tables" / "g2" / mode
            / f"{experiment_name}_s{reuse_params_from_seed}_grid.csv"
        )
        selection_method = "fixed_params_from_seed_grid"
    probabilities = aligned_probabilities(
        fitted_pipeline, test_x, class_count=len(dataset.classes)
    )

    prediction_path = root / "results" / "preds" / f"g2_{experiment_name}_{mode}_s{seed}.csv"
    _write_predictions(
        prediction_path,
        test_paths=test_paths,
        test_labels=test_y,
        probabilities=probabilities,
        classes=dataset.classes,
    )
    model_path = root / "models" / "g2" / mode / f"{experiment_name}_s{seed}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_name(model_path.name + ".tmp")
    artifact = {
        "format": "gemstone-trends-g2",
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": mode,
        "mask_selection": "class_label_conditioned_oracle" if mode == "published" else mode,
        "seed": seed,
        "feature_name": feature_name,
        "feature_length": feature_length,
        "classes": list(dataset.classes),
        "pipeline": fitted_pipeline,
        "best_params": best_params,
        "cv": {
            "folds": expected_folds,
            "scoring": scoring,
            "best_mean_neg_log_loss": best_cv_log_loss,
            "selection_method": selection_method,
            "tuning_source_seed": tuning_source_seed,
            "train_rows": int(len(train_x)),
            "feature_fingerprint": dataset.fingerprint,
        },
    }
    with temporary_model.open("wb") as stream:
        joblib.dump(artifact, stream, compress=3)
    os.replace(temporary_model, model_path)

    summary = {
        "model": model,
        "mode": mode,
        "mask_selection": "class_label_conditioned_oracle" if mode == "published" else mode,
        "seed": seed,
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "feature_name": feature_name,
        "feature_length": feature_length,
        "best_params": best_params,
        "best_cv_mean_log_loss": best_cv_log_loss,
        "selection_method": selection_method,
        "tuning_source_seed": tuning_source_seed,
        "prediction_file": prediction_path.relative_to(root).as_posix(),
        "model_file": model_path.relative_to(root).as_posix(),
        "grid_file": search_path.relative_to(root).as_posix(),
    }
    summary_path = (
        root / "results" / "tables" / "g2" / mode / f"{experiment_name}_s{seed}_summary.json"
    )
    _write_json(summary_path, summary)
    print(f"Tuning source seed {tuning_source_seed}; CV log loss: {best_cv_log_loss:.6f}")
    print(f"Predictions: {prediction_path.relative_to(root).as_posix()}")
    print(f"Model: {model_path.relative_to(root).as_posix()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    project = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"))
    parser.add_argument("--mode", choices=MASK_MODES)
    parser.add_argument("--feature", choices=FEATURE_VIEWS, default=FEATURE_NAME)
    parser.add_argument(
        "--allow-published-oracle", action="store_true",
        help="Permit the label-conditioned published-mask diagnostic comparison.",
    )
    parser.add_argument(
        "--reuse-params-from-seed", type=int,
        help="Reuse a completed seed's CV-selected parameters and only refit this seed.",
    )
    parser.add_argument("--seed", type=int, help="Run one seed; omit to run seeds 0 through 4.")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent grid-search jobs.")
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config = load_config(root / "configs" / "default.yaml")
    g2_config = config.get("g2", {})
    require_imblearn()
    requested_models = args.models or [str(g2_config.get("default_model", "rf"))]
    mode = args.mode or str(g2_config.get("default_mode", "labelfree"))
    if mode not in MASK_MODES:
        parser.error(f"Configured G2 default mode is invalid: {mode}")
    if mode == "published" and not args.allow_published_oracle:
        parser.error("--mode published requires explicit --allow-published-oracle.")
    if args.allow_published_oracle and mode != "published":
        parser.error("--allow-published-oracle is valid only with --mode published.")
    if args.reuse_params_from_seed is not None and args.seed is None:
        parser.error("--reuse-params-from-seed requires one explicit --seed.")
    if args.reuse_params_from_seed is not None and args.reuse_params_from_seed == args.seed:
        parser.error("The refit seed must differ from its tuning-source seed.")
    models = list(MODEL_NAMES) if "all" in requested_models else list(requested_models)
    seeds = [args.seed] if args.seed is not None else [
        int(value) for value in g2_config.get("seeds", [0, 1, 2, 3, 4])
    ]
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer.")
    if len(set(models)) != len(models):
        parser.error("Do not repeat a model name.")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Configured seed list must be non-empty and unique.")
    for model in models:
        for seed in seeds:
            train_one(
                root=root,
                model=model,
                mode=mode,
                seed=seed,
                jobs=args.jobs,
                refresh_features=args.refresh_features,
                feature_name=args.feature,
                allow_published_oracle=args.allow_published_oracle,
                reuse_params_from_seed=args.reuse_params_from_seed,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

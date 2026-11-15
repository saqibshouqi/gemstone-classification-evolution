# Gemstone Classification Evolution

Reproducible project comparing gemstone image classification approaches from
2015, 2022, and 2025. The common 2022 benchmark, data split, prediction format,
and evaluator are shared by all method packages.

## Project layout

    configs/                    Shared protocol and review thresholds
    data/manifests/             Selected source-file manifests (paths only)
    docs/                       Protocol, provenance, team contracts, AI-use log
    src/gemstone_trends/         Shared data and scoreboard code
    src/g1/ src/g2/ src/g3/     Stable method-package boundaries
    scripts/                    Thin command-line entry points
    splits/                     Fixed class order and canonical split CSV
    results/                    Generated predictions, tables, figures, audits
    models/                      Local/shared model weights; never committed
    notebooks/                   Exploration only; shared code lives under src/

The dataset directory is ignored by Git. The project stores only paths,
manifests, source code, and generated numeric results.

## Setup

Use Python 3.10 or newer. The pinned baseline requirements are in
requirements.txt; CNN, image-processing, and demo dependencies are split into
optional requirement files. From the project root:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

## Data and scoreboard

The raw Kaggle download has 87 classes. The canonical Chow benchmark uses 68
classes and the paper's selected 2,042/284 train/test files; filtering by class
alone would incorrectly retain 207 additional training images. The selected
file manifests are required to build the fixed split. The class-level
omissions and the 207 omitted retained-class training paths are recorded in
splits/excluded_classes.csv and splits/excluded_training_images.csv.

    python scripts/build_split.py --data-root ".\Gem Stone Images Dataset"
    python scripts/audit_duplicates.py
    python scripts/check_predictions.py results/preds/g2_rf_labelfree_s0.csv
    python scripts/evaluate.py

The duplicate audit emits candidate pairs and a contact sheet for manual
threshold review. Record the selected thresholds before reporting any
de-duplicated result. evaluate.py accepts any subset of prediction files, so
the scoreboard can be regenerated as packages deliver results.

For each final test prediction file, the evaluator writes both full-test and
test_no_dup scores. The latter drops test images whose high-confidence
duplicate group also contains a training image; the full-test result remains
available alongside it.

## G2: 2022 anchor

Install G2's class-balancing dependency, then train the Random Forest anchor on
the label-free Otsu selector. This writes the shared-format held-out test
prediction file, a reusable model artifact, and the fold-search table:

    python -m pip install -e ".[g2]"
    python -m g2.training --models rf --mode labelfree --seed 0
    python scripts/check_predictions.py results/preds/g2_rf_labelfree_s0.csv
    python scripts/evaluate.py results/preds/g2_rf_labelfree_s0.csv

Omit --seed to run the protocol seeds 0 through 4. Other classifier grids use
the same fixed features and fold-safe SMOTE:

    python -m g2.training --models lr lda knn dt nb svm --mode labelfree

See docs/g2_anchor.md for mask details, feature definitions, grid parameters,
and why the published per-class mask mode is diagnostic-only. G2 also provides
fixed gray and saturation controls; masks and feature caches are generated
locally and ignored by Git.

The completed seven-classifier comparison, E3 feature ablation, paper-reference
table, and duplicate-excluded results are summarized in
docs/g2_reproduction_report.md. Regenerate the report after evaluation with:

    python scripts/evaluate.py
    python scripts/render_g2_report.py

See docs/experiment_protocol.md for the split and prediction contracts,
docs/data_provenance.md for dataset limits, and docs/ai_usage_log.md for the
coursework production record.

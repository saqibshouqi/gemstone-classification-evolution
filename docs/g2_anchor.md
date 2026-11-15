# G2: the 2022 reproduction anchor

G2 is implemented as a self-contained arm in src/g2. It uses the shared
canonical split, class order, mask path convention, and prediction CSV
contract. The anchor feature is the normalized joint RGB histogram with 8 bins
per channel (512 values) concatenated with a 256-bin, 8-point/radius-1 circular
LBP histogram (768 values total). The RGB histogram counts foreground pixels
only; LBP covers the complete masked RGB image, including the zero-valued
segmentation background.

## Segmentation

The implementation provides both Otsu branches from the design:

- Gray intensity: Gaussian sigma 2, Otsu threshold, corner-based polarity,
  hole filling, disk closing radius 8, removal of components below 301 pixels,
  hole filling, 2x2 erosion, and final hole filling.
- HSV saturation: the same sequence with Gaussian sigma 5 and disk closing
  radius 9.

All masks are single-channel PNGs with values 0/255, stored under
masks/g2_<mode>/<split>/<class>/<stem>.png. Source size, modification time,
mode, and schema are embedded in the PNG metadata; stale masks are regenerated.
Feature matrices are cached under data/processed/g2/ and invalidated when the
split or source file metadata changes.

The class-conditioned published selector reproduces the paper's per-class
gray/saturation choice. It requires the known class label. The trainer permits
it only with the explicit `--allow-published-oracle` flag, records the result as
`class_label_conditioned_oracle`, and the inference API refuses these models.
These scores reproduce the paper's segmentation condition but contain test
label information and are diagnostic only. They must be reported separately
from leakage-free scores. The label-free
selector compares each candidate's worst corner foreground fraction, then its
average corner foreground fraction; ties fall back to saturation. This is a
deterministic operational rule for the design's four-corner selector.

## Training and prediction

SMOTE with k=4 is inside an imbalanced-learn pipeline, so it is fit only on
each GridSearchCV training fold and again on the complete training split when
refitting the selected model. The held-out test images are never oversampled or
used for model selection. The canonical training fold column drives a
PredefinedSplit; scoring is negative log loss. The prediction CSV contains
the held-out test rows only and uses the shared fixed probability-column order.
CV selection scores are saved separately; they are not presented as an
independent out-of-fold estimate because the same folds select hyperparameters.

The RF anchor is the default. LR, LDA, KNN, decision tree, Gaussian NB, and
one-vs-rest SVM grids are available through the same runner. Random seeds 0-4
are used by default. `--seed N` runs one seed; `--models all` runs every model.
By default, each seed repeats CV parameter selection. For long-running SVM
experiments, a completed grid can instead be reused for further stochastic
refits; `--reuse-params-from-seed N` verifies the source mode, feature schema,
and dataset fingerprint, then refits SMOTE and the SVM with that seed. Each
refit summary records the tuning source seed and points to its grid table.
The E3 core feature ablation reuses the cached descriptors with `--feature
rgb8`, `--feature lbp8_r1`, and `--feature rgb8_lbp8_r1`; no image is reprocessed
between feature views.

From the repository root, install the anchor dependency and train:

    python -m pip install -e ".[g2]"
    python -m g2.training --models rf --mode labelfree --seed 0
    python scripts/check_predictions.py results/preds/g2_rf_labelfree_s0.csv
    python scripts/evaluate.py results/preds/g2_rf_labelfree_s0.csv

To inspect fixed segmentation controls, use --mode saturation or --mode grey.
To generate masks without fitting a model:

    python -m g2.segmentation --modes saturation grey labelfree

To run all seven classifiers with the label-free segmentation comparison:

    python -m g2.training --models all --mode labelfree --jobs 4

To run one SVM grid per mask condition, then retain five stochastic SVM
refits using each condition's selected parameters:

    python -m g2.training --models svm --mode labelfree --seed 0 --jobs 8
    python -m g2.training --models svm --mode labelfree --seed 1 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode labelfree --seed 2 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode labelfree --seed 3 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode labelfree --seed 4 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode published --allow-published-oracle --seed 0 --jobs 8
    python -m g2.training --models svm --mode published --allow-published-oracle --seed 1 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode published --allow-published-oracle --seed 2 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode published --allow-published-oracle --seed 3 --reuse-params-from-seed 0 --jobs 1
    python -m g2.training --models svm --mode published --allow-published-oracle --seed 4 --reuse-params-from-seed 0 --jobs 1

The paper's per-class published condition and the three-way E3 feature
ablation are run explicitly as follows (the published results are oracle
diagnostics, not deployment estimates):

    python -m g2.training --models all --mode published --allow-published-oracle --jobs 4
    python -m g2.training --models rf --mode labelfree --feature rgb8 --jobs 4
    python -m g2.training --models rf --mode labelfree --feature lbp8_r1 --jobs 4
    python -m g2.training --models rf --mode published --allow-published-oracle --feature rgb8 --jobs 4
    python -m g2.training --models rf --mode published --allow-published-oracle --feature lbp8_r1 --jobs 4

Omit `--seed` to run all five planned seeds. Since the supplied dataset contains
the original raw images rather than the paper's saved `*_masked` images, both
conditions reconstruct masks from the design's documented Otsu recipe; the
published condition is therefore a recipe-level reproduction, not an exact
pixel-for-pixel reproduction of the authors' masks. The all-33-feature grid,
K-means/colour-space variants, Haralick features, and further histogram
combinations remain stretch experiments outside this core comparison.

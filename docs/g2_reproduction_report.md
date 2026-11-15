# G2: 2022 gemstone classification reproduction

## Scope and protocol

This report covers the core G2 package in the implementation design: both documented Otsu branches, the label-free mask selector, the seven classical classifiers, and E3's histogram-only / LBP-only / combined-feature ablation. The comparison uses the fixed 68-class Chow split (2,042 train and 284 test images), five predefined CV folds for parameter selection, SMOTE with k=4 inside the training pipeline, and five stochastic seeds for test-score aggregation.

The combined feature vector has 512 normalized joint RGB histogram values and 256 LBP values. The E3 views select the first 512 values, the last 256 values, or all 768. Grid selection uses negative log loss on the fixed five-fold split; the held-out test set is used only for final scoring.

**Mask modes:** `labelfree` selects between gray and saturation Otsu candidates using corner foreground occupancy and is the leakage-free result. `published` selects the branch using each image's known class label, so its scores are oracle diagnostics and must not be treated as deployable performance. The two modes are reported in separate rows throughout.

The supplied dataset contains raw source images, not the authors' saved `*_masked` files. Masks are reconstructed from the documented Otsu recipes, so this is a recipe-level reproduction rather than a pixel-exact reproduction of the original masks.

The SVM's full four-kernel / three-C / two-gamma grid was searched with five-fold CV once per mask mode using seed 0. Seeds 1–4 refit SMOTE and the classifier using that mode's saved CV-selected parameters. Their summaries record `fixed_params_from_seed_grid` and `tuning_source_seed: 0`; this avoids repeating the unusually expensive probability-calibrated OVR grid while retaining five stochastic test predictions per mode. The other model and feature-view runs performed their CV selection for each seed.

## Classifier comparison: full test set

Values are mean +/- standard deviation over seeds 0–4. Accuracy, macro-F1, and ECE are percentages; lower ECE is better.

| Model / feature view | Mask mode | N | Top-1 accuracy | Top-5 accuracy | Macro-F1 | ECE |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | Label-free | 284 | 65.56% +/- 1.48 | 93.94% +/- 0.76 | 64.01% +/- 1.70 | 41.64% +/- 1.65 |
| Random Forest | Published oracle | 284 | 68.17% +/- 1.31 | 94.93% +/- 0.64 | 66.74% +/- 1.22 | 44.58% +/- 0.95 |
| Logistic Regression | Label-free | 284 | 59.08% +/- 0.63 | 92.46% +/- 0.53 | 57.83% +/- 0.85 | 21.52% +/- 0.65 |
| Logistic Regression | Published oracle | 284 | 60.28% +/- 1.10 | 93.31% +/- 0.25 | 59.32% +/- 1.18 | 21.67% +/- 1.21 |
| One-vs-rest SVM | Label-free | 284 | 64.08% +/- 1.47 | 87.39% +/- 0.84 | 62.13% +/- 1.62 | 9.42% +/- 0.67 |
| One-vs-rest SVM | Published oracle | 284 | 64.44% +/- 1.51 | 86.34% +/- 1.04 | 63.07% +/- 1.39 | 9.25% +/- 1.68 |
| Gaussian Naive Bayes | Label-free | 284 | 63.31% +/- 0.68 | 80.70% +/- 0.39 | 62.20% +/- 0.66 | 36.60% +/- 0.60 |
| Gaussian Naive Bayes | Published oracle | 284 | 63.31% +/- 0.80 | 80.92% +/- 0.29 | 62.00% +/- 0.85 | 36.65% +/- 0.73 |
| Linear Discriminant Analysis | Label-free | 284 | 54.93% +/- 2.43 | 92.75% +/- 0.69 | 53.62% +/- 2.68 | 43.10% +/- 2.26 |
| Linear Discriminant Analysis | Published oracle | 284 | 56.55% +/- 1.97 | 94.08% +/- 0.29 | 55.94% +/- 1.97 | 41.46% +/- 1.54 |
| k-Nearest Neighbours | Label-free | 284 | 51.62% +/- 1.10 | 83.24% +/- 1.04 | 50.07% +/- 1.20 | 10.23% +/- 1.11 |
| k-Nearest Neighbours | Published oracle | 284 | 51.48% +/- 0.58 | 84.23% +/- 1.35 | 50.12% +/- 0.38 | 10.94% +/- 1.07 |
| Decision Tree | Label-free | 284 | 25.42% +/- 4.45 | 56.06% +/- 7.94 | 23.28% +/- 4.99 | 14.73% +/- 4.04 |
| Decision Tree | Published oracle | 284 | 25.49% +/- 5.19 | 56.97% +/- 8.05 | 22.83% +/- 5.35 | 14.38% +/- 3.40 |

## E3 feature ablation: full test set

| Model / feature view | Mask mode | N | Top-1 accuracy | Top-5 accuracy | Macro-F1 | ECE |
|---|---|---:|---:|---:|---:|---:|
| RGB histogram + LBP | Label-free | 284 | 65.56% +/- 1.48 | 93.94% +/- 0.76 | 64.01% +/- 1.70 | 41.64% +/- 1.65 |
| RGB histogram + LBP | Published oracle | 284 | 68.17% +/- 1.31 | 94.93% +/- 0.64 | 66.74% +/- 1.22 | 44.58% +/- 0.95 |
| RGB histogram only | Label-free | 284 | 64.58% +/- 1.07 | 94.15% +/- 0.53 | 62.81% +/- 1.10 | 39.64% +/- 1.23 |
| RGB histogram only | Published oracle | 284 | 64.08% +/- 1.17 | 94.72% +/- 0.56 | 62.61% +/- 1.09 | 39.13% +/- 1.04 |
| LBP only | Label-free | 284 | 21.34% +/- 1.49 | 41.13% +/- 0.58 | 18.71% +/- 1.55 | 10.46% +/- 1.80 |
| LBP only | Published oracle | 284 | 21.20% +/- 0.94 | 40.99% +/- 2.15 | 18.93% +/- 0.72 | 10.07% +/- 0.62 |

## Duplicate-excluded test set

The evaluator excludes the two test images marked by the frozen high-confidence duplicate audit, leaving 282 images. The full-test rows above remain the primary scores.

| Model / feature view | Mask mode | N | Top-1 accuracy | Top-5 accuracy | Macro-F1 | ECE |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | Label-free | 282 | 65.32% +/- 1.49 | 93.90% +/- 0.77 | 63.81% +/- 1.70 | 41.47% +/- 1.65 |
| Random Forest | Published oracle | 282 | 67.94% +/- 1.32 | 94.89% +/- 0.64 | 66.63% +/- 1.23 | 44.44% +/- 0.99 |
| Logistic Regression | Label-free | 282 | 59.15% +/- 0.63 | 92.41% +/- 0.54 | 57.89% +/- 0.87 | 21.68% +/- 0.65 |
| Logistic Regression | Published oracle | 282 | 60.35% +/- 1.10 | 93.26% +/- 0.25 | 59.51% +/- 1.18 | 21.70% +/- 1.06 |
| One-vs-rest SVM | Label-free | 282 | 64.18% +/- 1.48 | 87.30% +/- 0.85 | 62.35% +/- 1.64 | 9.59% +/- 0.68 |
| One-vs-rest SVM | Published oracle | 282 | 64.54% +/- 1.53 | 86.24% +/- 1.05 | 63.28% +/- 1.43 | 9.06% +/- 1.66 |
| Gaussian Naive Bayes | Label-free | 282 | 63.05% +/- 0.68 | 80.57% +/- 0.39 | 61.88% +/- 0.66 | 36.86% +/- 0.61 |
| Gaussian Naive Bayes | Published oracle | 282 | 63.05% +/- 0.81 | 80.78% +/- 0.30 | 61.69% +/- 0.86 | 36.91% +/- 0.73 |
| Linear Discriminant Analysis | Label-free | 282 | 55.11% +/- 2.29 | 92.70% +/- 0.69 | 53.64% +/- 2.44 | 42.95% +/- 2.13 |
| Linear Discriminant Analysis | Published oracle | 282 | 56.60% +/- 1.98 | 94.04% +/- 0.30 | 56.13% +/- 1.98 | 41.40% +/- 1.55 |
| k-Nearest Neighbours | Label-free | 282 | 51.99% +/- 1.11 | 83.12% +/- 1.05 | 50.42% +/- 1.21 | 9.87% +/- 1.13 |
| k-Nearest Neighbours | Published oracle | 282 | 51.56% +/- 0.54 | 84.11% +/- 1.36 | 50.30% +/- 0.46 | 10.84% +/- 1.03 |
| Decision Tree | Label-free | 282 | 25.53% +/- 4.39 | 55.96% +/- 7.86 | 23.40% +/- 4.99 | 14.59% +/- 4.11 |
| Decision Tree | Published oracle | 282 | 25.46% +/- 5.29 | 56.88% +/- 7.99 | 22.82% +/- 5.43 | 14.41% +/- 3.33 |
| RGB histogram only | Label-free | 282 | 64.40% +/- 1.14 | 94.11% +/- 0.54 | 62.73% +/- 1.16 | 39.55% +/- 1.30 |
| RGB histogram only | Published oracle | 282 | 63.83% +/- 1.18 | 94.68% +/- 0.56 | 62.52% +/- 1.09 | 38.95% +/- 1.08 |
| LBP only | Label-free | 282 | 20.78% +/- 1.50 | 40.71% +/- 0.58 | 17.71% +/- 1.55 | 9.97% +/- 1.83 |
| LBP only | Published oracle | 282 | 20.92% +/- 0.87 | 40.57% +/- 2.17 | 18.38% +/- 0.68 | 9.81% +/- 0.62 |

## Comparison with the paper reference

The design document records the paper's reported top-1 / top-5 reference values. The table compares them to this reproduction's leakage-free label-free results; the paper's values and our reconstructed-mask results are not expected to match exactly. The design also notes that the paper selected the best result among 231 candidates on its test set, which can make those reference values optimistic. This reproduction selected parameters only by the fixed training folds and never by held-out test scores.

| Classifier | Paper top-1 / top-5 | Reproduced label-free top-1 | Reproduced label-free top-5 |
|---|---:|---:|---:|
| Random Forest | 69.4% / 94.4% | 65.56% +/- 1.48 | 93.94% +/- 0.76 |
| Logistic Regression | 68.7% / 92.6% | 59.08% +/- 0.63 | 92.46% +/- 0.53 |
| One-vs-rest SVM | 66.9% / 86.3% | 64.08% +/- 1.47 | 87.39% +/- 0.84 |
| Gaussian Naive Bayes | 62.7% / 77.8% | 63.31% +/- 0.68 | 80.70% +/- 0.39 |
| Linear Discriminant Analysis | 59.9% / 94.0% | 54.93% +/- 2.43 | 92.75% +/- 0.69 |
| k-Nearest Neighbours | 54.6% / 85.9% | 51.62% +/- 1.10 | 83.24% +/- 1.04 |
| Decision Tree | 46.5% / 73.9% | 25.42% +/- 4.45 | 56.06% +/- 7.94 |

## Main observations

- Among leakage-free combined-feature classifiers, Random Forest has the highest mean top-1 accuracy (65.56%) and top-5 accuracy (93.94%).
- In the RF E3 ablation, RGB histogram only reaches 64.58% top-1, while LBP only reaches 21.34%; the combined view reaches 65.56%. This dataset result shows most of the RF discrimination comes from the color histogram, with the combined descriptor adding about 0.98 percentage points over histogram only.
- Published-oracle RF reaches 68.17% top-1. The gain over label-free RF is an expected diagnostic consequence of using the test image's class to choose its mask; it is not a fair operational performance estimate.
- The label-free RF score changes from 65.56% on all 284 test images to 65.32% on the 282-image duplicate-excluded subset.

## Reproducibility artifacts

- `results/tables/scoreboard.csv`: all 36 aggregate rows, including all metrics for full and duplicate-excluded test sets.
- `results/tables/per_seed_metrics.csv`: per-seed scores for every prediction file.
- `results/tables/g2/<mode>/`: ranked CV grid tables and per-seed summaries; SVM seed 1–4 summaries link to their mode's seed 0 grid.
- `results/preds/`: 90 validated, shared-contract prediction files.
- `results/figures/trend_<mode>_<test-set>.svg`: four G2 comparison charts.
- `results/tables/mcnemar_exact.csv`, `per_class_metrics.csv`, `risk_coverage.csv`, and `confusion/`: supporting evaluation outputs.
- `models/g2/<mode>/`: local refittable model artifacts, excluded from Git.

To regenerate the aggregates, run `python scripts/evaluate.py`; to regenerate this report from the scoreboard, run `python scripts/render_g2_report.py`.

## Scope boundary

The 33-feature-method grid is marked stretch work in the design. This reproduction completes the core E3 three-way histogram/LBP ablation, not the stretch grid or the other 2015/2025 method packages.

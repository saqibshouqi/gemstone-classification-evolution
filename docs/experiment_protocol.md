# Shared experiment protocol

This project compares the historical 2015 colour-statistics method, the 2022
feature/classifier method, and the 2025 fine-tuned CNN methods on one fixed
gemstone benchmark. The 2022 result is the reproduction anchor. The 2015 and
2025 papers use different datasets, so their methods are transferred to this
benchmark; their published accuracies are not direct targets.

## Data boundary

- Use the fixed 68-class order in `splits/classes.txt`.
- Use the authors' selected train and test filenames in the two manifests under
  `data/manifests/`; the expected counts are 2,042 train and 284 test images.
- Keep the original test membership unchanged. Do not tune on test predictions.
- Audit train/test and within-training near-duplicate candidates before model
  comparisons. Assign stratified five-fold CV folds to training rows only after
  that audit, keeping high-confidence duplicate groups in one fold. This adds
  a leakage-control constraint to the paper's stratified CV protocol.
- All stochastic model runs use seeds 0 through 4 and report mean and standard
  deviation.
- Preserve the source images locally; this repository stores paths and code,
  not the image data.

## Prediction contract

Each prediction CSV is named
`<arm>_<model>_<mode>_s<seed>.csv` and has these leading columns:

`path,true,where,<one probability column per class in classes.txt order>`

`where` is `test` or `oof`; rows contain relative image paths, the ground-truth
class, and a normalized probability distribution. If a model has no probability
output, use normalized vote shares. `check_predictions.py` validates paths,
labels, class order, finite values, row sums, and split membership.

## Scoreboard

The evaluator reports top-1 and top-5 accuracy, macro-F1, per-class recall,
confusion matrices, 95% bootstrap intervals (1,000 resamples), exact paired
McNemar tests, expected calibration error, and risk-coverage rows. It also
aggregates metrics across seeds. Test rows are scored only by the evaluator.

The duplicate cutoffs in `configs/default.yaml` are initial review thresholds,
not established findings. Inspect `results/audits/near_duplicate_review.jpg`,
adjust the thresholds based on the flagged pairs, and record the final choices
and review date before using the de-duplicated comparison. The audit currently
uses a lightweight HOG plus RGB histogram descriptor as its visual embedding;
this can be replaced by a pretrained CNN embedding later without changing the
pair-file contract.

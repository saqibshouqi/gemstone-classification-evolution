# G1 (2015 method) handoff

Status on 20 Nov 2026. Method: Sinkevicius, Lipnickas and Rimkus (2015), rebuilt on the gemstone benchmark. Code is in `src/g1/`, how it maps to the paper is in `docs/g1_implementation.md`. Read that file's "Decisions the paper does not settle" section before changing anything.

## Where things stand

The G1 pipeline is complete and runs end to end: stone mask, grey + HSV, 400-pixel rings, 16 statistics per ring, decision tree with ring voting, committees (random subsets and Half&Half), prediction files, and the `api.py` adapter. Unit tests pass (`python -m tests.test_g1`).

**The numbers below were produced on a STAND-IN split, not the team's split.** The stand-in used every image in the repo's `train_selected` (2,042) and `test_selected` (284) folders, with 5 folds I assigned myself, no duplicate handling and no excluded classes or images. The team's `splits/chow68.csv` has exclusions and duplicate handling, so these numbers are NOT comparable with the team's tables. They are kept only as a reference for what to expect. Everything must be rerun on the team's split (see "To do").

Masks: published masks were built from the repo's `*_masked` images by matching them to the originals by pixel content (file names differ, e.g. `Alexandrite0.jpg` vs `alexandrite_1.jpg`). One test image, `almandine_28.jpg`, had no usable mask and got a whole-image mask.

### Stand-in split results (published masks)

5-fold CV on the training images (`ladder`):

| Model | CV accuracy |
|---|---|
| all pixels, one tree | 48.3% |
| all pixels, pruned tree | 48.0% |
| 400-px rings, one tree | 61.5% |
| 16 trees, random subsets | 66.7% |
| 16 trees, Half&Half | 66.8% |

Ring-size sweep (CV): all pixels 48.3, 3200 px 53.3, 1600 px 58.7, 800 px 61.0, 400 px 61.5, 200 px 64.2.

Committee sweep (CV), trees 1 / 2 / 4 / 8 / 16 / 32: random 57.6 / 62.1 / 66.4 / 67.0 / 66.7 / 66.9, Half&Half 61.5 / 62.7 / 66.1 / 66.6 / 66.8 / 67.1.

Test set (`final`, mean of 3 seeds, 284 images, 95% interval about +/- 5.5 points):

| Model | top-1 | top-5 | macro-F1 |
|---|---|---|---|
| all pixels, one tree | 48.2% | 48.2% | 0.480 |
| all pixels, pruned tree | 45.5% | 45.5% | 0.449 |
| 400-px rings, one tree | 58.9% | 89.6% | 0.579 |
| 16 trees, random subsets | 62.9% | 94.5% | 0.605 |
| 16 trees, Half&Half | 63.5% | 94.3% | 0.608 |

What this showed: the ladder order (all pixels < rings < committee) holds, smaller rings help with diminishing returns, random subsets match Half&Half, and pruning did not help (the pruned tree was unstable across seeds). The paper's own data (30 amber colour classes) is not public, so G1 is validated by method fidelity and trend agreement, not by matching its 81.6%.

## To do (in this order)

1. Clone this branch, create a venv, `python -m pip install -r requirements-g1.txt`, run `python -m tests.test_g1` (5 PASS lines).
2. Get the gem data into `data/chow68` (not committed, about 575 MB, no licence stated): `git clone --depth 1 https://github.com/hybchow/gems data/chow68`. If the clone stalls on a slow connection, download the zip from `https://github.com/hybchow/gems/archive/refs/heads/master.zip` with `curl.exe -L -C -` and rename the extracted folder to `data/chow68`.
3. Check that the team's `splits/chow68.csv` `path` column matches what `src/g1/data_io.py` expects (paths relative to `data/chow68`, e.g. `train_selected/Amber/amber_1.jpg`). Fix `load_split` or the path handling if not. Do NOT run anything that overwrites the team's split: `prepare` keeps an existing split file.
4. Masks: if the team has a script that produces `masks/published/<split>/<class>/<stem>.png`, use it so all three arms share identical masks. Otherwise `python -m src.g1.run_g1 prepare --data-root data/chow68` builds them (several minutes, prints progress). Expect a few images without a confident match; they get a whole-image mask and are listed in the output.
5. Rerun on the team's split, published masks:
   - `ladder` (about 15 min), `ringsweep` (a few min), `committees` (about 30 min). The Half&Half member is the slow part.
   - `final --mode published --seeds 0 1 2 3 4` (30 to 60 min). Run it only once the settings are fixed: it reads the test set. Add `--oof` only if the team needs out-of-fold rows (it repeats the cross-validation for every seed).
6. Label-free masks: when the team's `masks/labelfree` exists, run `final --mode labelfree`. Until then `--mode labelfree-g1seg` uses my approximate stand-in segmenter (follows the 2022 paper's description, not the authors' notebook). Label it as an approximation if used.
7. Validate with `python scripts/check_predictions.py`, get official numbers from `python scripts/evaluate.py`, and move G1 tables to `results/tables/g1/` to match G2's layout. Delete or relabel the stand-in tables in `results/tables/g1_standin_split/` once the real ones exist.
8. Add a G1 entry to `docs/ai_usage_log.md` if the coursework requires declaring AI help.

## Known issues and notes

- `prepare` (content matching) is slow and was only validated on the repo's original files and on synthetic data.
- Output filenames look like `g1_hh16-rings400_published_s0.csv` (model names use hyphens, no underscores, so `<arm>_<model>_<mode>_s<seed>` splits cleanly).
- `segment_fallback.py` is imported by `data_io.py` even when you don't use label-free mode. Don't delete it.
- The pruned tree (`pdt-allpix`) picks its pruning strength by cross-validation, which varies by seed; it did worse than the plain tree. Report it as a result.
- Not implemented: the paper's colour-space comparison (HSV + grey vs RGB or grey only). Optional extra.
- Settings the paper does not specify (stone rescaled to about 15,000 px, ring ordering, Half&Half out-of-bag handling, 50% subsets) are listed in `docs/g1_implementation.md`. Keep them unchanged for the final runs.
- `src/g1/metrics.py` is a helper. Official numbers come from `scripts/evaluate.py`.
- Cross-validation was about 3 points more optimistic than the test set on the stand-in split.

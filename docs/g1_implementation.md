# G1: 2015 rebuild (Sinkevicius, Lipnickas and Rimkus), package C

Rebuilds the 2015 amber colour classifier on the team's shared 68-class gemstone split:
stone mask -> grey + HSV -> 400-pixel concentric rings -> 16 colour statistics per ring ->
decision tree on rings -> majority vote -> committee of trees (random subsets or Half&Half).

**This is a method re-run, not a reproduction.** The paper used 30 amber colour classes, 8,479
samples, a white belt with ring lighting and 2-fold CV. We keep the method and change the data.
Report the shape of the ladder, never "we reproduced 81.60%".

## Install and run (from the repo root)

```
pip install -r requirements.txt
python -m tests.test_g1                                   # 5 unit checks, ~2 s

python -m src.g1.run_g1 prepare    --data-root data/chow68   # only until package A's split file exists
python -m src.g1.run_g1 ladder     --mode published          # rungs a, b, c with 5-fold CV
python -m src.g1.run_g1 ringsweep  --mode published          # all pixels, 3200, 1600, 800, 400, 200
python -m src.g1.run_g1 committees --mode published          # 1 to 32 trees, random vs Half&Half
python -m src.g1.run_g1 final      --mode published --seeds 0 1 2 3 4 --oof   # test run, once
python -m src.g1.run_g1 final      --mode labelfree --seeds 0 1 2 3 4 --oof   # when B's masks exist
```

Until package B's label-free masks exist, use `--mode labelfree-g1seg` (masks computed on the fly by
`segment_fallback.py`). Rename the prediction files / re-run with `--mode labelfree` once B's masks land.

Outputs: `results/g1/*.csv` (tables, each with a `.meta.json` of args, library versions, git hash),
`results/preds/g1_<model>_<mode>_s<seed>.csv` (contract 3), `models/g1-<model>_<mode>.joblib`
(for `src/g1/api.py`, contract 4).

## What maps to the paper

| Step | Where | Paper |
|---|---|---|
| Stone mask, drop background | `rings.py` | Sec. 2.2 |
| Grey + H, S, V | `rings.pixel_table` | Sec. 3.1 (RGB, LAB tried, rejected) |
| Rings of equal pixel count from the centre, leftover dropped | `rings.make_rings` | Sec. 2.4, Fig. 6 |
| mean, std, kurtosis, skewness (16 numbers) | `features.py` | Eq. 1 to 5 |
| Tree on rings, vote over rings | `committee.py` | Sec. 2.4 |
| Committee 1 to 32, random subsets / Half&Half | `committee.py` | Sec. 3.1, Fig. 8 |
| Ladder: all pixels -> rings -> committee | `run_g1.py ladder` | 69.29 -> ~80 -> 81.60 (amber) |
| Ring-size sweep | `run_g1.py ringsweep` | Fig. 7 |

## Decisions the paper does not settle (write these in the report)

1. **Rescale to ~15,000 stone pixels** (crop to the bounding box first). Rings are counted in pixels and our images are 89 to 3,947 px wide; the paper's stones were 3,000 to 30,000 px, peak near 15,000. `--target-area` changes it.
2. **Ring order**: stone pixels sorted by distance from the mask centroid (ties by angle), cut into consecutive 400-pixel groups. The paper only says "rings of equal pixel count from the centre".
3. **Statistics**: std divides by M-1 (Eq. 2); kurtosis and skewness use population moments (MATLAB default); flat rings get skew 0, kurtosis 3.
4. **Raw hue** as published; `--circular-hue` adds sin/cos hue (optional extra).
5. **Random-subset members** train on 50% of the stones (`--subset-frac`).
6. **Half&Half out-of-bag trap**: a fully grown tree gets its own training stones almost all right. Member 1 sees all stones, so it has no out-of-bag stones; we use a 5-fold stone-level CV vote of single trees as the starting estimate, and every later member adds votes only on stones it did not train on. `committees` also reports the out-of-bag accuracy after each member. If Half&Half underperforms, report the random-subset committee (the paper shows it is competitive).
7. **PDT** = cost-complexity pruning with `ccp_alpha` picked by 3-fold stone-level CV inside the training data.
8. CV folds come from the split file's `fold` column and are by stone, never by ring.
9. Vote shares are the "probabilities" in the prediction files (many exact zeros). `top5` counts a hit only if the true class received at least one vote; log-loss uses additive smoothing (1e-3).

## Check on the real data before trusting numbers

- `prepare` assumes `train_selected/<Class>/*.jpg` and `train_masked/<Class>/*.png` style folders in `hybchow/gems`. If the real layout differs, edit `discover_images` / `derive_published_masks` in `data_io.py`. Counts must come out at 2,042 train and 284 test.
- If the repo's masked images are JPEG, the black background may not be exactly 0: use `prepare --mask-threshold 10`.
- The log line "images that fell back to a whole-image mask" should be 0.
- `segment_fallback.py` follows the description of the 2022 pipeline, not the authors' notebook. Package B's selector replaces it.
- Everything here was tested only on synthetic data (`tests/make_synthetic.py`); the real accuracy is unknown. Expect it far below the paper's 81.60% (68 varieties, web photos, 24 to 44 training images per class), and below G2's 69.4%. That gap is hypothesis H1, a finding rather than a failure.
- Do not run `final` casually: it reads the test split.

"""G1 (2015, Sinkevicius et al.) rebuilt on the shared gemstone split.

Run from the repository root:

  python -m src.g1.run_g1 prepare   --data-root data/chow68
  python -m src.g1.run_g1 ladder    --mode published
  python -m src.g1.run_g1 ringsweep --mode published
  python -m src.g1.run_g1 committees --mode published
  python -m src.g1.run_g1 final     --mode published --seeds 0 1 2 3 4 --oof

Model names (no underscores, because the prediction file name uses them as separators):
  dt-allpix, pdt-allpix            rung (a): one 16-number vector from all stone pixels
  dt-rings400, pdt-rings400        rung (b): 400-pixel rings with voting
  random16-rings400, hh16-rings400 rung (c): committee of 16 trees (random subsets / Half&Half)
Replace 16 / 400 with any other number.

Cross-validation uses the `fold` column of the split file (train rows only). The test
split is only read by `final`; do not run `final` casually.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import metrics as M
from .committee import RingCommittee, RingData
from .data_io import (FeatureStore, build_split, derive_published_masks, load_split, size_key)

SPEC_RE = re.compile(r"^(?P<kind>dt|pdt|random(?P<nr>\d+)|hh(?P<nh>\d+))-(?P<ring>allpix|rings(?P<size>\d+))$")
LADDER = [("a", "dt-allpix"), ("a", "pdt-allpix"), ("b", "dt-rings400"),
          ("c", "random16-rings400"), ("c", "hh16-rings400")]
SWEEP_SIZES = ["all", 3200, 1600, 800, 400, 200]
CHECKPOINTS = [1, 2, 4, 8, 16, 32]


# ------------------------------------------------------------------ model specs
def parse_spec(name: str) -> dict:
    m = SPEC_RE.match(name)
    if not m:
        raise ValueError(f"bad model name '{name}' (examples: dt-rings400, hh16-rings400)")
    kind = m["kind"]
    if kind.startswith("random"):
        sel, n = "random", int(m["nr"])
    elif kind.startswith("hh"):
        sel, n = "halfhalf", int(m["nh"])
    else:
        sel, n = "all", 1
    ring = None if m["ring"] == "allpix" else int(m["size"])
    return dict(name=name, selection=sel, n=n, pruned=(kind == "pdt"), ring=ring)


def build_model(spec: dict, seed: int, args) -> RingCommittee:
    return RingCommittee(n_trees=spec["n"], selection=spec["selection"], subset_frac=args.subset_frac,
                         ccp_alpha="auto" if spec["pruned"] else 0.0,
                         class_weight=args.class_weight, seed=seed)


# ------------------------------------------------------------------ run metadata
def show(d):
    print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()})


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def save_meta(args, name):
    import cv2, scipy, sklearn
    out = Path(args.results) / "g1"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(command=args.command, args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                opencv=cv2.__version__, git_commit=_git_hash(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ data helpers
def load_everything(args):
    df, classes = load_split(args.splits, args.classes)
    cidx = {c: i for i, c in enumerate(classes)}
    df["y"] = df["class"].map(cidx)
    store = FeatureStore(args.data_root, args.mask_root, args.cache, args.target_area,
                         args.circular_hue, args.n_jobs)
    return df, classes, store


def _data(feats, idx, y, k):
    return RingData.from_lists([feats[i] for i in idx], y[idx], k)


def cv_run(feats, y, folds, k, spec, seed, args):
    """Stone-level k-fold CV on the training images -> out-of-fold vote shares."""
    oof = np.zeros((len(y), k))
    for f in sorted(set(folds)):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        model = build_model(spec, seed, args).fit(_data(feats, tr, y, k))
        oof[va] = model.predict_proba(_data(feats, va, y, k))
    return oof


def fold_scores(oof, y, folds):
    accs, lls = [], []
    for f in sorted(set(folds)):
        va = folds == f
        accs.append(M.top1(oof[va], y[va]))
        lls.append(M.log_loss(oof[va], y[va]))
    return np.array(accs), np.array(lls)


def _train_part(df):
    tr = df[df.split == "train"].reset_index(drop=True)
    return tr, tr["y"].to_numpy(), tr["fold"].to_numpy()


# ------------------------------------------------------------------ commands
def cmd_prepare(args):
    split_csv, classes_txt = Path(args.splits), Path(args.classes)
    if split_csv.exists():
        print(f"split file exists, keeping it: {split_csv}")
        df, _ = load_split(split_csv, classes_txt)
    else:
        df = build_split(args.data_root, split_csv, classes_txt, seed=0)
        print(f"STAND-IN split written to {split_csv} (replace with package A's file when it exists)")
    print(df.groupby("split").size().to_string())
    made, missing, worst = derive_published_masks(args.data_root, df, args.mask_root, args.mask_threshold)
    print(f"published masks derived from the repo's *_masked images (matched by content): "
          f"{made} written, {missing} without a confident match, worst accepted pixel difference {worst:.1f}")


def cmd_ladder(args):
    df, classes, store = load_everything(args)
    tr, y, folds = _train_part(df)
    k = len(classes)
    sizes = sorted({parse_spec(n)["ring"] or 0 for _, n in LADDER}, key=lambda s: -1 if s == 0 else s)
    sizes = [None if s == 0 else s for s in sizes]
    feats_by = store.get(tr, args.mode, sizes)
    rows = []
    for rung, name in LADDER:
        spec = parse_spec(name)
        t0 = time.time()
        oof = cv_run(feats_by[size_key(spec["ring"])], y, folds, k, spec, args.seed, args)
        acc, ll = fold_scores(oof, y, folds)
        rows.append(dict(rung=rung, model=name, cv_acc=acc.mean(), cv_acc_std=acc.std(ddof=1),
                         cv_logloss=ll.mean(), oof_top5=M.topk(oof, y), oof_macro_f1=M.macro_f1(oof, y),
                         seconds=round(time.time() - t0, 1)))
        show(rows[-1])
    out = pd.DataFrame(rows)
    _save(args, f"ladder_{args.mode}", out)


def cmd_ringsweep(args):
    df, classes, store = load_everything(args)
    tr, y, folds = _train_part(df)
    feats_by = store.get(tr, args.mode, SWEEP_SIZES)
    rows = []
    for s in SWEEP_SIZES:
        spec = dict(name=f"dt-{s}", selection="all", n=1, pruned=False, ring=None if s == "all" else s)
        feats = feats_by[size_key(s)]
        oof = cv_run(feats, y, folds, len(classes), spec, args.seed, args)
        acc, ll = fold_scores(oof, y, folds)
        rows.append(dict(ring_size=s, rings_per_stone=float(np.mean([len(f) for f in feats])),
                         cv_acc=acc.mean(), cv_acc_std=acc.std(ddof=1), cv_logloss=ll.mean()))
        show(rows[-1])
    _save(args, f"ringsweep_{args.mode}", pd.DataFrame(rows))


def cmd_committees(args):
    df, classes, store = load_everything(args)
    tr, y, folds = _train_part(df)
    k = len(classes)
    feats = store.get(tr, args.mode, [args.ring_size])[size_key(args.ring_size)]
    rows = []
    for sel in ("random", "halfhalf"):
        per_fold = {n: [] for n in CHECKPOINTS}
        oob_hist = []
        for f in sorted(set(folds)):
            a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            model = RingCommittee(n_trees=max(CHECKPOINTS), selection=sel, subset_frac=args.subset_frac,
                                  class_weight=args.class_weight, seed=args.seed).fit(_data(feats, a, y, k))
            dva = _data(feats, b, y, k)
            staged = model.staged_proba(dva, CHECKPOINTS)
            for n in CHECKPOINTS:
                per_fold[n].append(M.top1(staged[n], y[b]))
            if model.oob_acc_:
                oob_hist.append(model.oob_acc_)
        for n in CHECKPOINTS:
            v = np.array(per_fold[n])
            rows.append(dict(selection=sel, n_trees=n, cv_acc=v.mean(), cv_acc_std=v.std(ddof=1),
                             hh_oob_acc=float(np.mean([h[n - 1] for h in oob_hist])) if oob_hist else np.nan))
            show(rows[-1])
    _save(args, f"committees_{args.mode}", pd.DataFrame(rows))


def write_preds(path, paths, true_names, proba, where, classes):
    out = pd.DataFrame(proba, columns=classes)
    out.insert(0, "where", where)
    out.insert(0, "true", true_names)
    out.insert(0, "path", paths)
    out.to_csv(path, index=False, float_format="%.6f")


def cmd_final(args):
    df, classes, store = load_everything(args)
    k = len(classes)
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    ytr, yte, folds = tr["y"].to_numpy(), te["y"].to_numpy(), tr["fold"].to_numpy()
    specs = [parse_spec(n) for n in args.models]
    rings = sorted({s["ring"] or 0 for s in specs}, key=lambda s: -1 if s == 0 else s)
    rings = [None if s == 0 else s for s in rings]
    ftr, fte = store.get(tr, args.mode, rings), store.get(te, args.mode, rings)
    preds_dir = Path(args.results) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        key = size_key(spec["ring"])
        for seed in args.seeds:
            model = build_model(spec, seed, args).fit(_data(ftr[key], np.arange(len(tr)), ytr, k))
            dte = _data(fte[key], np.arange(len(te)), yte, k)
            p_te = model.predict_proba(dte)
            parts = [(te["path"], te["class"], p_te, ["test"] * len(te))]
            if args.oof:
                oof = cv_run(ftr[key], ytr, folds, k, spec, seed, args)
                parts.insert(0, (tr["path"], tr["class"], oof, ["oof"] * len(tr)))
            write_preds(preds_dir / f"g1_{spec['name']}_{args.mode}_s{seed}.csv",
                        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                        np.vstack([p[2] for p in parts]), sum([p[3] for p in parts], []), classes)
            correct = (p_te.argmax(1) == yte).astype(float)
            lo, hi = M.bootstrap_ci(correct)
            rows.append(dict(model=spec["name"], mode=args.mode, seed=seed, **M.summary(p_te, yte),
                             ci_low=lo, ci_high=hi, alpha=getattr(model, "alpha_", np.nan)))
            show(rows[-1])
            if seed == args.seeds[0]:
                joblib.dump(dict(model=model, classes=classes, ring_size=spec["ring"], target_area=args.target_area,
                                 circular_hue=args.circular_hue, name=spec["name"], mode=args.mode),
                            Path(args.models_dir) / f"g1-{spec['name']}_{args.mode}.joblib")
    out = pd.DataFrame(rows)
    _save(args, f"final_{args.mode}", out)
    agg = out.groupby("model")[["top1", "top5", "macro_f1"]].agg(["mean", "std"])
    print(agg.round(4).to_string())
    print(f"images that fell back to a whole-image mask (mask missing or tiny): {store.n_fallback}")


def _save(args, name, df):
    out = Path(args.results) / "g1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    save_meta(args, name)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nwritten: {out / (name + '.csv')}")


# ------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "ladder", "ringsweep", "committees", "final"])
    p.add_argument("--data-root", default="data/chow68")
    p.add_argument("--splits", default="splits/chow68.csv")
    p.add_argument("--classes", default="splits/classes.txt")
    p.add_argument("--mask-root", default="masks")
    p.add_argument("--cache", default="cache/g1")
    p.add_argument("--results", default="results")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--mode", default="published", help="published | labelfree | labelfree-g1seg")
    p.add_argument("--target-area", type=int, default=15000)
    p.add_argument("--circular-hue", action="store_true")
    p.add_argument("--subset-frac", type=float, default=0.5)
    p.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    p.add_argument("--ring-size", type=int, default=400, help="for `committees`")
    p.add_argument("--seed", type=int, default=0, help="for CV commands")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="for `final`")
    p.add_argument("--models", nargs="+", default=["dt-allpix", "pdt-allpix", "dt-rings400",
                                                  "random16-rings400", "hh16-rings400"], help="for `final`")
    p.add_argument("--oof", action="store_true", help="`final`: also write out-of-fold rows for the train split")
    p.add_argument("--mask-threshold", type=int, default=10, help="`prepare`: stone = max channel > this")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"prepare": cmd_prepare, "ladder": cmd_ladder, "ringsweep": cmd_ringsweep,
     "committees": cmd_committees, "final": cmd_final}[args.command](args)


if __name__ == "__main__":
    main()

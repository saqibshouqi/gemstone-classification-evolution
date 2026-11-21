"""Split file, masks and cached ring features for G1.

Contracts used (from the team design doc):
  1. splits/chow68.csv with columns path, class, split (train|test), fold (0-4, train
     rows only), dup_group. `path` is relative to the data root.
  2. masks/<mode>/<split>/<class>/<stem>.png, values 0 or 255. <mode> is "published"
     or "labelfree". Extra G1-only mode "labelfree-g1seg" computes masks on the fly
     with segment_fallback.py (use it only until package B's real masks exist).

Until package A publishes the real split file, `build_split` makes a stand-in from the
hybchow/gems folders train_selected and test_selected (ASSUMPTION: they contain one
sub-folder per class; adjust discover_images if the real layout differs). Never
regenerate the split once the team's file exists.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from .features import ring_features
from .rings import (MIN_MASK_PIXELS, TARGET_AREA, clean_mask, crop_and_scale, pixel_table,
                    read_image, read_mask)
from .segment_fallback import label_free_mask

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ON_THE_FLY_MODES = {"labelfree-g1seg"}


# --------------------------------------------------------------------------- split
def discover_images(data_root: Path, folder: str):
    base = Path(data_root) / folder
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found")
    rows = []
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            if p.parent == base:
                raise ValueError(f"{p} sits directly in {folder}/; expected <class>/ sub-folders. "
                                 "Adjust discover_images for the real layout.")
            rows.append((p.relative_to(data_root).as_posix(), p.parent.name))
    return rows


def build_split(data_root, out_csv, classes_txt, n_folds=5, seed=0) -> pd.DataFrame:
    recs = []
    for folder, split in (("train_selected", "train"), ("test_selected", "test")):
        for rel, cls in discover_images(data_root, folder):
            recs.append(dict(path=rel, **{"class": cls}, split=split, fold=-1, dup_group=""))
    df = pd.DataFrame(recs)
    classes = sorted(df["class"].unique())
    tr = df.split == "train"
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    folds = np.full(tr.sum(), -1)
    ytr = df.loc[tr, "class"].values
    for f, (_, va) in enumerate(skf.split(np.zeros(len(ytr)), ytr)):
        folds[va] = f
    df.loc[tr, "fold"] = folds
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["fold"] = out["fold"].where(out["fold"] >= 0, other=pd.NA)
    out.to_csv(out_csv, index=False)
    Path(classes_txt).write_text("\n".join(classes) + "\n")
    return df


def load_split(csv_path, classes_txt):
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").fillna(-1).astype(int)
    classes = [c for c in Path(classes_txt).read_text().splitlines() if c.strip()]
    missing = set(df["class"]) - set(classes)
    if missing:
        raise ValueError(f"classes in split but not in classes.txt: {sorted(missing)[:5]}")
    return df, classes


# --------------------------------------------------------------------------- masks
def mask_path(mask_root, mode, split, cls, path_rel) -> Path:
    return Path(mask_root) / mode / split / cls / (Path(path_rel).stem + ".png")


def _small(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def _pair_cost(sel_small, msk_small, thr=20):
    """A masked image is the original with the background blacked out, so on the stone
    pixels the two must agree. Median absolute difference over those pixels (about 2 to 6
    for a true pair, far higher for a wrong one)."""
    fg = msk_small.max(axis=2) > thr
    if fg.sum() < 20:
        return 1e3
    return float(np.median(np.abs(sel_small[fg] - msk_small[fg])))


def derive_published_masks(data_root, df, mask_root, threshold=10, max_cost=15.0):
    """Stand-in for package B's masks, derived from the repo's *_masked images.

    The repo's masked files are NOT named like the originals (e.g. Alexandrite0.jpg vs
    alexandrite_1.jpg), so each original is paired with a masked file of the same class
    by CONTENT (optimal one-to-one assignment on a pixel-difference cost). Stone = max
    channel above `threshold` (10 because the masked files are JPEG, so the black
    background is not exactly 0). Returns (written, unmatched, worst_cost).
    """
    from scipy.optimize import linear_sum_assignment

    data_root = Path(data_root)
    made = unmatched = 0
    worst = 0.0
    bad = []
    groups = list(df.groupby(["split", "class"]))
    for n_done, ((split, cls), g) in enumerate(groups, start=1):
        print(f"\r  matching masks: class group {n_done}/{len(groups)} ({split}/{cls})        ", end="", flush=True)
        mdir = data_root / f"{split}_masked"
        folder = mdir / cls
        if not folder.is_dir() and mdir.is_dir():
            lower = {d.name.lower(): d for d in mdir.iterdir() if d.is_dir()}
            folder = lower.get(cls.lower(), folder)
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT) if folder.is_dir() else []
        sel_paths = list(g["path"])
        if not files:
            unmatched += len(sel_paths)
            bad += sel_paths
            continue
        sel_small = [_small(data_root / p) for p in sel_paths]
        msk_small = [_small(p) for p in files]
        cost = np.full((len(sel_paths), len(files)), 1e3)
        for i, sm in enumerate(sel_small):
            for j, mm in enumerate(msk_small):
                if sm is not None and mm is not None:
                    cost[i, j] = _pair_cost(sm, mm)
        rows, cols = linear_sum_assignment(cost)
        done = set()
        for i, j in zip(rows, cols):
            if cost[i, j] > max_cost:
                continue
            worst = max(worst, float(cost[i, j]))
            img = cv2.imread(str(files[j]), cv2.IMREAD_COLOR)
            dst = mask_path(mask_root, "published", split, cls, sel_paths[i])
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), (img.max(axis=2) > threshold).astype(np.uint8) * 255)
            made += 1
            done.add(i)
        for i, p in enumerate(sel_paths):
            if i not in done:
                unmatched += 1
                bad.append(p)
    print()
    if bad:
        # No usable masked partner (e.g. the repo's mask is blank). The authors kept every
        # test image whatever its mask looked like, so we do too: a whole-image mask, which
        # is also what G1 falls back to for an empty mask. Listed so it can be reported.
        for p in bad:
            sp, cl = p.split("/")[0].replace("_selected", ""), p.split("/")[1]
            im = cv2.imread(str(data_root / p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            dst = mask_path(mask_root, "published", sp, cl, p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), np.full(im.shape[:2], 255, np.uint8))
        print(f"WHOLE-IMAGE fallback mask written for {len(bad)} image(s): {bad[:10]}")
    return made, unmatched, worst


# ------------------------------------------------------------------ feature cache
def size_key(s):
    return "all" if s in (None, "all") else int(s)


def _extract_one(img_path, msk_path, mode, sizes, target_area, circ):
    img = read_image(img_path)
    mask = None
    if mode in ON_THE_FLY_MODES:
        mask = label_free_mask(img)
    elif msk_path is not None:
        mask = read_mask(msk_path)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
    fallback = False
    if mask is not None:
        mask = clean_mask(mask)
    if mask is None or mask.sum() < MIN_MASK_PIXELS:
        mask, fallback = np.ones(img.shape[:2], bool), True
    img2, mask2 = crop_and_scale(img, mask, target_area)
    table = pixel_table(img2, mask2, circ)
    return {size_key(s): ring_features(table, s) for s in sizes}, fallback


class FeatureStore:
    """Computes ring features for (mode, ring size) once and caches them on disk."""

    def __init__(self, data_root, mask_root, cache_dir, target_area=TARGET_AREA,
                 circular_hue=False, n_jobs=-1):
        self.data_root, self.mask_root = Path(data_root), Path(mask_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_area, self.circ, self.n_jobs = target_area, circular_hue, n_jobs
        self.n_fallback = 0

    def _file(self, mode, key):
        return self.cache_dir / f"feat_{mode}_a{self.target_area}_c{int(self.circ)}_r{key}.joblib"

    def get(self, df: pd.DataFrame, mode: str, sizes):
        keys = [size_key(s) for s in sizes]
        cache = {k: (joblib.load(self._file(mode, k)) if self._file(mode, k).exists() else {}) for k in keys}
        todo_keys = [k for k in keys if any(p not in cache[k] for p in df["path"])]
        if todo_keys:
            rows = [(p, c, sp) for p, c, sp in zip(df["path"], df["class"], df["split"])
                    if any(p not in cache[k] for k in todo_keys)]
            if mode not in ON_THE_FLY_MODES:
                paths = [mask_path(self.mask_root, mode, sp, c, p) for p, c, sp in rows]
                gone = [p for p in paths if not p.exists()]
                if gone:
                    raise FileNotFoundError(
                        f"{len(gone)} mask files missing for mode '{mode}', e.g. {gone[0]}. "
                        "Run `prepare` (published masks) or use --mode labelfree-g1seg.")
            else:
                paths = [None] * len(rows)
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_extract_one)(self.data_root / rel, mp, mode, todo_keys, self.target_area, self.circ)
                for (rel, _, _), mp in zip(rows, paths))
            for (rel, _, _), (feats, fb) in zip(rows, res):
                self.n_fallback += int(fb)
                for k in todo_keys:
                    cache[k][rel] = feats[k]
            for k in todo_keys:
                joblib.dump(cache[k], self._file(mode, k))
        return {k: [cache[k][p] for p in df["path"]] for k in keys}
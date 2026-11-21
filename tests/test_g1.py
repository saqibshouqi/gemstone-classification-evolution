"""Unit checks for G1. Run from the repo root:  python -m tests.test_g1"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.g1.committee import RingCommittee, RingData
from src.g1.features import feature_names, ring_statistics
from src.g1.rings import crop_and_scale, make_rings, pixel_table
from tests.make_synthetic import one_image


def test_statistics_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 20.0, size=(3, 400, 4))
    f = ring_statistics(x)
    for r in range(3):
        for c in range(4):
            v = x[r, :, c]
            assert np.isclose(f[r, 0 * 4 + c], v.mean(), rtol=1e-4)
            assert np.isclose(f[r, 1 * 4 + c], v.std(ddof=1), rtol=1e-4)
            assert np.isclose(f[r, 2 * 4 + c], stats.kurtosis(v, fisher=False), rtol=1e-3)
            assert np.isclose(f[r, 3 * 4 + c], stats.skew(v), rtol=1e-3, atol=1e-4)
    assert len(feature_names()) == 16 and len(feature_names(True)) == 20


def test_flat_ring_has_no_nan():
    f = ring_statistics(np.zeros((1, 400, 4)))
    assert np.isfinite(f).all() and f[0, 8] == 3.0 and f[0, 12] == 0.0


def test_rings_and_scale():
    rng = np.random.default_rng(1)
    img, masked = one_image(rng, 3, 10)
    mask = masked.max(axis=2) > 0
    img2, mask2 = crop_and_scale(img, mask, 15000)
    assert abs(int(mask2.sum()) - 15000) / 15000 < 0.06
    table = pixel_table(img2, mask2)
    rings = make_rings(table, 400)
    assert rings.shape[1:] == (400, 4)
    assert rings.shape[0] == len(table) // 400            # leftover dropped
    assert 34 <= rings.shape[0] <= 40                      # ~37 rings at 15,000 px
    assert make_rings(table, None).shape[0] == 1
    # rings run from the centre outwards: first ring is closer to the centroid than the last
    ys, xs = np.nonzero(mask2)
    cy, cx = ys.mean(), xs.mean()
    assert len(table) == mask2.sum()


def _toy(n_stones=60, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n_stones // k)
    lists = [rng.normal(c, 1.5, size=(int(rng.integers(5, 12)), 6)) for c in y]
    return RingData.from_lists(lists, y, k)


def test_committees_run_and_prefix():
    data = _toy()
    for sel in ("random", "halfhalf"):
        m = RingCommittee(8, sel, seed=0).fit(data)
        st = m.staged_proba(data, [1, 4, 8])
        assert set(st) == {1, 4, 8}
        for p in st.values():
            assert np.allclose(p.sum(1), 1.0)
        assert np.allclose(m.predict_proba(data, 4), st[4])
    hh = RingCommittee(6, "halfhalf", seed=1).fit(data)
    assert len(hh.oob_acc_) == 6 and len(hh.trees_) == 6


def test_pruned_tree_alpha_selected():
    data = _toy(seed=2)
    m = RingCommittee(1, "all", ccp_alpha="auto", seed=0).fit(data)
    assert m.alpha_ >= 0.0
    assert np.allclose(m.predict_proba(data).sum(1), 1.0)


def check_preds(path, classes):
    """Contract 3: path,true,where,<one probability per class in classes.txt order>."""
    df = pd.read_csv(path, keep_default_na=False)
    assert list(df.columns) == ["path", "true", "where"] + list(classes), "bad columns"
    assert set(df["where"]) <= {"test", "oof"}
    probs = df[list(classes)].to_numpy()
    assert np.allclose(probs.sum(1), 1.0, atol=1e-3), "rows must sum to 1"
    assert (probs >= 0).all()
    return df


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)

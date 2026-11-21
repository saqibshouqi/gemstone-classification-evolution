"""Decision tree on rings, stone-level voting, and tree committees.

Paper (Sec. 2.4 and 3.1):
  * one decision tree is trained on rings, each ring carrying its stone's class;
  * a stone is classified by letting every ring vote (and, for a committee, every
    member vote on every ring); votes are summed, the class with most votes wins;
  * committee members are chosen two ways: random training subsets, or Half&Half
    (Breiman 1998): tree 1 sees all stones, each next tree sees half of the stones the
    current committee gets right plus half of those it gets wrong.

Our decisions where the paper is silent:
  * random-subset members each train on `subset_frac` (default 0.5) of the stones,
    drawn without replacement;
  * Half&Half "right/wrong" is decided out-of-bag. A fully grown tree fits its own
    training stones almost perfectly, so asking the committee about stones it trained
    on returns almost no errors. Tree 1 trains on all stones, so it has no true
    out-of-bag stones; we give the first step a surrogate: a stone-level k-fold
    cross-validated vote of single trees. Every later member adds its votes only on
    the stones it did NOT train on. Each member counts with weight 1 per stone
    (its vote shares are normalised), so the estimate is a plain out-of-bag committee.
  * folds are always split by STONE, never by ring, so rings of one stone cannot leak
    across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ALPHA_GRID = (0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)


@dataclass
class RingData:
    """All rings of a set of stones, stacked."""
    X: np.ndarray        # (N, F) ring features
    stone: np.ndarray    # (N,) stone index of every ring
    y: np.ndarray        # (S,) class index per stone (-1 if unknown)
    n_classes: int

    @classmethod
    def from_lists(cls, rings_list, y, n_classes):
        counts = [len(r) for r in rings_list]
        X = np.concatenate(rings_list).astype(np.float32)
        stone = np.repeat(np.arange(len(rings_list)), counts)
        return cls(X, stone, np.asarray(y, dtype=int), int(n_classes))

    @property
    def n_stones(self) -> int:
        return len(self.y)

    def rows(self, stones) -> np.ndarray:
        sel = np.zeros(self.n_stones, dtype=bool)
        sel[np.asarray(stones, dtype=int)] = True
        return sel[self.stone]


def _fit_tree(data: RingData, stones, seed, ccp_alpha=0.0, class_weight=None):
    rows = data.rows(stones)
    tree = DecisionTreeClassifier(random_state=seed, ccp_alpha=ccp_alpha, class_weight=class_weight)
    return tree.fit(data.X[rows], data.y[data.stone[rows]])


def _votes(tree, data: RingData) -> np.ndarray:
    """Ring votes of one tree per stone -> (S, K) integer counts."""
    k = data.n_classes
    pred = tree.predict(data.X)
    flat = data.stone * k + pred
    return np.bincount(flat, minlength=data.n_stones * k).reshape(data.n_stones, k)


def _splitter(y, k, seed):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.min() >= k:
        return StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return KFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)))


class RingCommittee:
    """selection: "all" (one tree on all stones), "random" or "halfhalf".

    ccp_alpha: 0.0 (unpruned tree, DT), a float, or "auto" (pruned tree, PDT: alpha is
    chosen by 3-fold stone-level cross-validation inside the training data).
    """

    def __init__(self, n_trees=1, selection="all", subset_frac=0.5, ccp_alpha=0.0,
                 class_weight=None, seed=0, hh_folds=5):
        if selection == "all" and n_trees != 1:
            raise ValueError("selection='all' means a single tree")
        if selection not in ("all", "random", "halfhalf"):
            raise ValueError(selection)
        self.n_trees, self.selection, self.subset_frac = n_trees, selection, subset_frac
        self.ccp_alpha, self.class_weight, self.seed, self.hh_folds = ccp_alpha, class_weight, seed, hh_folds

    # ------------------------------------------------------------------ fitting
    def _choose_alpha(self, data: RingData) -> float:
        best, best_acc = 0.0, -1.0
        folds = list(_splitter(data.y, 3, self.seed))
        for a in ALPHA_GRID:
            accs = []
            for tr, va in folds:
                tree = _fit_tree(data, tr, self.seed, a, self.class_weight)
                sub = _subset(data, va)
                accs.append((_votes(tree, sub).argmax(1) == sub.y).mean())
            acc = float(np.mean(accs))
            if acc >= best_acc:          # ties go to the larger alpha (smaller tree)
                best, best_acc = a, acc
        return best

    def _cv_oob_shares(self, data: RingData) -> np.ndarray:
        """Surrogate out-of-bag votes for Half&Half member 1 (see module docstring)."""
        shares = np.zeros((data.n_stones, data.n_classes))
        for tr, va in _splitter(data.y, self.hh_folds, self.seed):
            tree = _fit_tree(data, tr, self.seed, self.alpha_, self.class_weight)
            v = _votes(tree, _subset(data, va)).astype(float)
            shares[va] = v / np.maximum(v.sum(1, keepdims=True), 1)
        return shares

    def fit(self, data: RingData):
        s_all = np.arange(data.n_stones)
        rng = np.random.default_rng(self.seed)
        self.alpha_ = self._choose_alpha(data) if self.ccp_alpha == "auto" else float(self.ccp_alpha)
        self.trees_, self.oob_acc_ = [], []
        self.n_classes_ = data.n_classes

        if self.selection == "all":
            self.trees_.append(_fit_tree(data, s_all, self.seed, self.alpha_, self.class_weight))
            return self

        if self.selection == "random":
            size = max(1, int(round(self.subset_frac * data.n_stones)))
            for m in range(self.n_trees):
                pick = rng.choice(s_all, size=size, replace=False)
                self.trees_.append(_fit_tree(data, pick, self.seed + m, self.alpha_, self.class_weight))
            return self

        # Half&Half
        self.trees_.append(_fit_tree(data, s_all, self.seed, self.alpha_, self.class_weight))
        oob = self._cv_oob_shares(data)
        self.oob_acc_.append(_oob_acc(oob, data.y))
        for m in range(1, self.n_trees):
            ok = (oob.sum(1) > 0) & (oob.argmax(1) == data.y)
            right, wrong = np.flatnonzero(ok), np.flatnonzero(~ok)
            parts = []
            for grp in (right, wrong):
                if len(grp):
                    parts.append(rng.choice(grp, size=max(1, len(grp) // 2), replace=False))
            pick = np.concatenate(parts)
            tree = _fit_tree(data, pick, self.seed + m, self.alpha_, self.class_weight)
            self.trees_.append(tree)
            in_bag = np.zeros(data.n_stones, dtype=bool)
            in_bag[pick] = True
            v = _votes(tree, data).astype(float)
            v = v / np.maximum(v.sum(1, keepdims=True), 1)
            oob[~in_bag] += v[~in_bag]
            self.oob_acc_.append(_oob_acc(oob, data.y))
        return self

    # --------------------------------------------------------------- prediction
    def staged_proba(self, data: RingData, checkpoints):
        """Vote-share matrices using the first n members, for each n in checkpoints."""
        want = set(checkpoints)
        cum = np.zeros((data.n_stones, data.n_classes))
        out = {}
        for i, tree in enumerate(self.trees_, start=1):
            cum += _votes(tree, data)
            if i in want:
                out[i] = cum / cum.sum(1, keepdims=True)
        return out

    def predict_proba(self, data: RingData, n_use: int | None = None) -> np.ndarray:
        n = len(self.trees_) if n_use is None else n_use
        return self.staged_proba(data, [n])[n]


def _subset(data: RingData, stones) -> RingData:
    stones = np.asarray(stones, dtype=int)
    rows = data.rows(stones)
    remap = -np.ones(data.n_stones, dtype=int)
    remap[stones] = np.arange(len(stones))
    return RingData(data.X[rows], remap[data.stone[rows]], data.y[stones], data.n_classes)


def _oob_acc(oob, y) -> float:
    has = oob.sum(1) > 0
    return float((oob.argmax(1)[has] == y[has]).mean()) if has.any() else float("nan")

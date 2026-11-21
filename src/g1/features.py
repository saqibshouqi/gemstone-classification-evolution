"""Per-ring colour statistics (paper Eq. 1 to 5).

For each ring: mean, standard deviation, kurtosis and skewness of grey, H, S and V,
in the paper's order (Eq. 5): [mu_G, mu_HSV, sigma_G, sigma_HSV, k_G, k_HSV, s_G, s_HSV]
which is 16 numbers (20 with the optional circular hue).

Conventions (the paper does not spell all of them out):
  * standard deviation divides by M - 1 (Eq. 2);
  * kurtosis and skewness use population moments E[(v-mu)^k] / sigma^k with the
    population sigma (this is MATLAB's default, which the authors used); kurtosis is
    the plain fourth moment (3 for a normal curve);
  * a perfectly flat ring (sigma = 0, e.g. black pixels) gets skewness 0 and
    kurtosis 3 instead of NaN.
"""
from __future__ import annotations

import numpy as np

from .rings import make_rings

STATS = ("mean", "std", "kurt", "skew")


def channel_names(circular_hue: bool = False) -> list[str]:
    return ["G", "Hsin", "Hcos", "S", "V"] if circular_hue else ["G", "H", "S", "V"]


def feature_names(circular_hue: bool = False) -> list[str]:
    ch = channel_names(circular_hue)
    return [f"{s}_{c}" for s in STATS for c in ch]


def ring_statistics(rings: np.ndarray) -> np.ndarray:
    """(R, M, C) -> (R, 4*C) float32."""
    x = rings.astype(np.float64)
    m = x.shape[1]
    mu = x.mean(axis=1)
    d = x - mu[:, None, :]
    d2 = d ** 2
    std = np.sqrt(d2.sum(axis=1) / max(m - 1, 1))             # Eq. 2, ddof = 1
    sd_pop = np.sqrt(d2.mean(axis=1))
    flat = sd_pop < 1e-9
    safe = np.where(flat, 1.0, sd_pop)
    kurt = (d2 ** 2).mean(axis=1) / safe ** 4                   # Eq. 3, fisher = False
    skew = (d ** 3).mean(axis=1) / safe ** 3                    # Eq. 4
    kurt[flat] = 3.0
    skew[flat] = 0.0
    return np.concatenate([mu, std, kurt, skew], axis=1).astype(np.float32)


def ring_features(table: np.ndarray, ring_size: int | None) -> np.ndarray:
    """Sorted pixel table -> (R, F) ring feature matrix."""
    return ring_statistics(make_rings(table, ring_size))

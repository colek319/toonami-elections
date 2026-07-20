"""Election strategies.

Each takes (past, noms) — two show x dimension profile DataFrames — and
returns a distance Series keyed by nomination. Lower = less surprise.
"""

from itertools import combinations

import numpy as np
import pandas as pd


def euclidean(past, noms):
    """Euclidean distance from the taste centroid (mean of past profiles).

    Every dimension counts equally, regardless of how much past shows
    actually vary along it.
    """
    centroid = past.mean(axis=0)
    return np.sqrt(((noms - centroid) ** 2).sum(axis=1))


def portfolio(past, noms, size=3):
    """Score the best *set* of 3 rather than the 3 individually closest.

    A lineup's distance is how far its *average* profile lands from the
    taste centroid, and each nomination scores the distance of the best
    lineup containing it. Two shows that miss the centroid in opposite
    directions cancel out, so a balanced slate can beat three shows that
    each hug the centroid alone.
    """
    centroid = past.mean(axis=0)
    size = min(size, len(noms.index))
    best = pd.Series(np.inf, index=noms.index)
    for combo in map(list, combinations(noms.index, size)):
        d = np.sqrt(((noms.loc[combo].mean() - centroid) ** 2).sum())
        best[combo] = np.minimum(best[combo], d)
    return best


def mahalanobis(past, noms):
    """Mahalanobis distance from the taste centroid.

    Distance is scaled by the covariance of past show profiles: a miss
    along a dimension the club's past picks rarely vary in (say, Rad is
    always high) costs more than a miss along one that's all over the
    place. Uses the pseudo-inverse in case the covariance is singular.
    """
    centroid = past.mean(axis=0)
    vi = np.linalg.pinv(np.cov(past.to_numpy(), rowvar=False))
    delta = (noms - centroid).to_numpy()
    return pd.Series(
        np.sqrt(np.einsum("ij,jk,ik->i", delta, vi, delta)),
        index=noms.index,
    )

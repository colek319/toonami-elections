"""Election strategies.

Each takes (past, noms) — two show x dimension profile DataFrames — and
returns a distance Series keyed by nomination. Lower = less surprise.
"""

import numpy as np
import pandas as pd


def euclidean(past, noms):
    """Euclidean distance from the taste centroid (mean of past profiles).

    Every dimension counts equally, regardless of how much past shows
    actually vary along it.
    """
    centroid = past.mean(axis=0)
    return np.sqrt(((noms - centroid) ** 2).sum(axis=1))


def mahalanobis(past, noms):
    """Mahalanrbis distance from the taste centroid.

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

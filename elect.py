"""Toonami July 2026 — vote wizard election script.

Watchers scored every show from Toonami past on six dimensions
(Goon, Cute, Laugh, Edgy, Rad, Aesthetic), 1-5. The shows already
voted in over past seasons define the club's "taste centroid" — an
average point in 6-D dimension space. The election picks the 3
nominations whose profiles land closest to that point.

MINIMIZE FREE ENERGY: the winning shows are the least surprising
given what the club has historically watched.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "toonami-jul-2026.csv"

DIMENSIONS = ["Goon", "Cute", "Laugh", "Edgy", "Rad", "Aesthetic"]

# Shows up for nomination this season (names as they appear in the CSV).
NOMINATIONS = [
    "Jaadugar: A Witch In Mongolia",
    "The 100 Girlfriends Who Really Really Really Really REALLY Love You season 3",
    "You and I Are Polar Opposites season 2",
    "Sparks of Tomorrow",
    "Smoking Behind the Supermarket With You",
    "Chainsmoker Cat",
    "Let's Go Kaikigumi",
]

COLUMN_RE = re.compile(r"^(?P<show>.+) \[(?P<dim>" + "|".join(DIMENSIONS) + r")\]$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path=CSV_PATH):
    """Parse the wide Google Forms CSV into a voter x (show, dim) DataFrame.

    Rows are indexed by voter name; columns are a MultiIndex of
    (show, dimension). Show names are whitespace-normalized (the export
    has e.g. "Ranma 1/2  S2"), and non-rating columns (Timestamp, Email
    Address, junk like "[Row 7]") are dropped. Missing votes stay NaN.
    """
    raw = pd.read_csv(path)
    raw = raw.set_index(raw["Name"].str.strip())

    keep, keys = [], []
    for col in raw.columns:
        m = COLUMN_RE.match(col.strip())
        if m:
            keep.append(col)
            keys.append((" ".join(m.group("show").split()), m.group("dim")))

    df = raw[keep].astype(float)
    df.columns = pd.MultiIndex.from_tuples(keys, names=["show", "dim"])
    return df


# ---------------------------------------------------------------------------
# Imputation strategies
# ---------------------------------------------------------------------------
# Each takes the raw ratings DataFrame and returns a copy with every NaN
# filled in (in memory — the CSV is never touched).

def impute_column_mean(df):
    """Fill a missing cell with the crowd's mean for that (show, dimension).

    Falls back to the voter's own overall mean, then the scale midpoint.
    Simple, but assumes a missing vote looks like the crowd's vote and
    ignores that some voters rate harsh and some rate generous.
    """
    out = df.fillna(df.mean())                              # column mean
    out = out.apply(lambda row: row.fillna(df.loc[row.name].mean()), axis=1)
    return out.fillna(3.0)                                  # last resort

def impute_voter_bias(df):
    """Column mean, shifted by how harsh or generous the voter rates overall.

    Fill = column_mean + (voter_mean - global_mean), clipped to the 1-5
    scale. A generous rater's missing votes are imputed high, a harsh
    rater's low. Falls back to the plain column mean where a voter has no
    observed votes at all.
    """
    bias = df.mean(axis=1) - np.nanmean(df.to_numpy())
    fill = pd.DataFrame(
        np.add.outer(bias.fillna(0.0).to_numpy(), df.mean().to_numpy()),
        index=df.index,
        columns=df.columns,
    ).clip(1, 5)
    out = df.fillna(fill)
    return out.fillna(3.0)                                  # columns nobody rated


# ---------------------------------------------------------------------------
# Election strategies
# ---------------------------------------------------------------------------
# Each takes (past, noms) — two show x dimension profile DataFrames — and
# returns a free-energy Series keyed by nomination. Lower = less surprise.

def euclidean(past, noms):
    """Euclidean distance from the taste centroid (mean of past profiles).

    Every dimension counts equally, regardless of how much past shows
    actually vary along it.
    """
    centroid = past.mean(axis=0)
    return np.sqrt(((noms - centroid) ** 2).sum(axis=1))

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


# ---------------------------------------------------------------------------
# Election
# ---------------------------------------------------------------------------

def profiles(df):
    """Each show's mean score per dimension -> show x dimension DataFrame."""
    return df.mean(axis=0).unstack("dim")[DIMENSIONS]


class Election:
    """One season's election: how to impute, what to load, who's running, how to score.

    >>> Election(impute_voter_bias, load, NOMINATIONS, mahalanobis).run()
    """

    def __init__(self, imputer, loader, targets, strategy, n=3):
        self.imputer = imputer
        self.loader = loader
        self.targets = list(targets)
        self.strategy = strategy
        self.n = n

    def ranking(self):
        """Free energy for every target, lowest (least surprising) first."""
        prof = profiles(self.imputer(self.loader()))
        past = prof.drop(index=self.targets)
        noms = prof.loc[self.targets]
        return self.strategy(past, noms).sort_values()

    def run(self):
        """The n winners as a Series of (show -> free energy)."""
        return self.ranking().head(self.n)


# ---------------------------------------------------------------------------

def main():
    elections = {
        "column-mean impute + Euclidean": Election(
            impute_column_mean, load, NOMINATIONS, euclidean
        ),
        "voter-bias impute + Mahalanobis": Election(
            impute_voter_bias, load, NOMINATIONS, mahalanobis
        ),
    }

    for name, election in elections.items():
        print(f"=== {name} ===")
        print(election.ranking().round(3).to_string())
        print("winners:", ", ".join(election.run().index), "\n")


if __name__ == "__main__":
    main()

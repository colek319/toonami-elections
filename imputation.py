"""Imputation strategies.

Each takes the raw voter x (show, dim) ratings DataFrame and returns a copy
with every NaN filled in (in memory — the CSV is never touched).
"""

import numpy as np
import pandas as pd


def impute_column_mean(df):
    """Fill a missing cell with the crowd's mean for that (show, dimension).

    Falls back to the voter's own overall mean, then the scale midpoint.
    Simple, but assumes a missing vote looks like the crowd's vote and
    ignores that some voters rate harsh and some rate generous.
    """
    out = df.fillna(df.mean())                              # column mean
    out = out.apply(lambda row: row.fillna(df.loc[row.name].mean()), axis=1)
    return out.fillna(3.0)                                  # last resort


def impute_knn(df, k=3):
    """Fill from the k voters whose ballots correlate most with this one.

    A missing cell becomes the mean of what the voter's k nearest
    neighbours — by Pearson correlation over the columns both ballots
    rated — gave that (show, dimension). Anti-correlated voters are
    never neighbours. Cells no neighbour rated fall back to the column
    mean, then the scale midpoint.
    """
    corr = df.T.corr(min_periods=2)
    out = df.copy()
    for voter in df.index:
        missing = df.columns[df.loc[voter].isna()]
        sims = corr.loc[voter].drop(voter).dropna()
        neighbours = sims[sims > 0].nlargest(k).index
        if len(missing) and len(neighbours):
            out.loc[voter, missing] = df.loc[neighbours, missing].mean()
    return out.fillna(df.mean()).fillna(3.0)


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

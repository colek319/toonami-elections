"""Loaders.

Each is constructed with a file path and returns the processed ratings
when called with no arguments, so an Election can be handed
``Load("data/toonami-jul-2026.csv")`` and call it when it runs.
"""

from pathlib import Path

import pandas as pd

from constants import COLUMN_RE, CSV_PATH


class Load:
    """Parse a wide Google Forms CSV into a voter x (show, dim) DataFrame.

    Rows are indexed by voter name; columns are a MultiIndex of
    (show, dimension). Show names are whitespace-normalized (the export
    has e.g. "Ranma 1/2  S2"), and non-rating columns (Timestamp, Email
    Address, junk like "[Row 7]") are dropped. Missing votes stay NaN.
    """

    def __init__(self, path=CSV_PATH):
        # A relative path resolves next to the code, not the cwd, so
        # `python elect.py` works from anywhere.
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path(__file__).parent / self.path

    def __call__(self):
        raw = pd.read_csv(self.path)
        raw = raw.set_index(raw["Name"].str.strip())

        keep, keys = [], []
        for col in raw.columns:
            m = COLUMN_RE.match(col.strip())
            if m:
                keep.append(col)
                keys.append((" ".join(m.group("show").split()), m.group("dim")))

        df = raw[keep].astype(float)
        df.columns = pd.MultiIndex.from_tuples(keys, names=["show", "dim"])

        # Every show must have at least one rating; a show nobody rated is
        # dropped rather than imputed from nothing.
        rated = df.notna().sum().groupby(level="show").sum()
        return df.loc[:, df.columns.get_level_values("show").isin(rated[rated > 0].index)]

    def __repr__(self):
        return f"Load({str(self.path)!r})"

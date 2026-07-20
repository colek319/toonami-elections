# Toonami Vote Wizard

Seasonal election tooling for the watch club. Watchers rate every show from
Toonami past on six 1–5 dimensions (Goon, Cute, Laugh, Edgy, Rad, Aesthetic)
via a Google Form. Past voted-in shows define the club's taste centroid; the
election picks the 3 nominations closest to it.

## Running

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python elect.py
```

Prints the full ranking and the 3 winners for each configured election.

## Testing

```sh
pytest
```

`test_elect.py` covers the loader quirks (junk columns, whitespace
normalization, missing votes, dropping unrated shows), every imputer (fills
everything, keeps observed votes, never mutates the input; kNN filling from
the correlated neighbour), every strategy (hand-computed Euclidean
distances; Mahalanobis penalizing a miss on a low-variance dimension where
Euclidean can't tell the difference; portfolio electing a complementary
slate), and an end-to-end election plus a smoke test on the real season
CSV. Add a test alongside any new imputer or strategy — the parametrized
contract tests are the pattern to copy.

In a new shell later, just `source venv/bin/activate` again — the venv and
installed packages persist.

## Files

- `elect.py` — the `Election` class and the `main()` entry point.
- `loaders.py` — `Load`, the Google Forms CSV parser.
- `imputation.py` — imputation strategies.
- `election_strategies.py` — election strategies.
- `constants.py` — dimensions, nominations, and the CSV path.
- `test_elect.py` — the test suite.
- `data/toonami-jul-2026.csv` — this season's Google Forms export. Empty cells are
  unwatched/skipped shows; they get imputed in memory, never written back.

## Anatomy of an election

```python
Election(imputer, loader, targets, strategy, n=3)
```

| Arg | What it is | Existing options |
|---|---|---|
| `imputer` | fills missing votes | `impute_column_mean`, `impute_voter_bias`, `impute_knn` |
| `loader` | reads the CSV | `Load("data/toonami-jul-2026.csv")` — construct with a path (relative paths resolve next to the code); calling it returns the processed ratings |
| `targets` | shows up for vote | `SUMMER_2026_NOMINATIONS` |
| `strategy` | scores the targets | `euclidean`, `mahalanobis`, `portfolio` |

`.ranking()` gives the full sorted table; `.run()` gives the top `n` winners.

## Adding a new method

**New imputation** — add a function in `imputation.py` with this contract:
takes the raw voter × (show, dim) DataFrame from the loader (NaN = missing
vote), returns a copy with every NaN filled. Never mutate the input or touch
the CSV.

```python
def impute_matrix_factorization(df):
    """Fill from a low-rank reconstruction of the ratings matrix."""
    ...
    return filled_df
```

**New election strategy** — add a function in `election_strategies.py` with
this contract: takes `(past, noms)`, two show × dimension profile DataFrames
(past shows and nominations, already voter-averaged), returns a Series
indexed like `noms` where **lower = better**. If your method naturally
produces a "higher is better" score, negate it.

```python
def cosine(past, noms):
    """Angle from the taste centroid instead of distance to it."""
    ...
    return scores  # pd.Series indexed by nomination, lower wins
```

**Wire it up** — add an entry to the `elections` dict in `main()`:

```python
"matrix impute + cosine": Election(
    impute_matrix_factorization, Load("data/toonami-jul-2026.csv"), SUMMER_2026_NOMINATIONS, cosine
),
```

Run `python elect.py` and compare it against the existing methods side by
side.

## New season checklist

1. Export the Google Form results to `data/toonami-<mon>-<year>.csv` and
   point `CSV_PATH` in `constants.py` at it.
2. Freeze the slate in `constants.py` as `<SEASON>_<YEAR>_NOMINATIONS`,
   names exactly as they appear in the CSV — a typo fails fast with a
   `ValueError` naming the bad shows.
3. Update the `elections` dict in `main()` to the new loader + nominations.
4. Run `python elect.py`, compare methods, and announce the official result.
5. Append the official structure to the table below. Leave old seasons'
   constants and CSVs in place — that's what keeps them reproducible.

## Past elections

The structure each season's official result was produced with, so any past
event can be reproduced.

| Season | Election structure |
|---|---|
| Summer 2026 | `Election(impute_voter_bias, Load("data/toonami-jul-2026.csv"), SUMMER_2026_NOMINATIONS, mahalanobis, n=3)` — "voter-bias impute + Mahalanobis" |

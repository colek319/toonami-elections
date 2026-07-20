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
normalization, missing votes, dropping unrated shows), both imputers (fills
everything, keeps observed votes, never mutates the input), both strategies
(hand-computed Euclidean distances; Mahalanobis penalizing a miss on a
low-variance dimension where Euclidean can't tell the difference), and an
end-to-end election plus a smoke test on the real season CSV. Add a test
alongside any new imputer or strategy — the parametrized contract tests are
the pattern to copy.

In a new shell later, just `source venv/bin/activate` again — the venv and
installed packages persist.

## Files

- `elect.py` — loading, imputation strategies, election strategies, and the
  `Election` class.
- `toonami-jul-2026.csv` — this season's Google Forms export. Empty cells are
  unwatched/skipped shows; they get imputed in memory, never written back.
- `research.md` — notes on alternative imputation and election methods.

## Anatomy of an election

```python
Election(imputer, loader, targets, strategy, n=3)
```

| Arg | What it is | Existing options |
|---|---|---|
| `imputer` | fills missing votes | `impute_column_mean`, `impute_voter_bias` |
| `loader` | reads the CSV | `Load("toonami-jul-2026.csv")` — construct with a path (bare filenames resolve next to the code); calling it returns the processed ratings |
| `targets` | shows up for vote | `NOMINATIONS` |
| `strategy` | scores the targets | `euclidean`, `mahalanobis` |

`.ranking()` gives the full sorted table; `.run()` gives the top `n` winners.

## Adding a new method

**New imputation** — add a function under *Imputation strategies* with this
contract: takes the raw voter × (show, dim) DataFrame from the loader (NaN =
missing vote), returns a copy with every NaN filled. Never mutate the input
or touch the CSV.

```python
def impute_knn(df):
    """Fill from the k voters whose ballots correlate most with this one."""
    ...
    return filled_df
```

**New election strategy** — add a function under *Election strategies* with
this contract: takes `(past, noms)`, two show × dimension profile DataFrames
(past shows and nominations, already voter-averaged), returns a Series
indexed like `noms` where **lower = better**. If your method naturally
produces a "higher is better" score, negate it.

```python
def portfolio(past, noms):
    """Score the best *set* of 3 rather than the 3 individually closest."""
    ...
    return scores  # pd.Series indexed by nomination, lower wins
```

**Wire it up** — add an entry to the `elections` dict in `main()`:

```python
"knn impute + portfolio": Election(impute_knn, load, NOMINATIONS, portfolio),
```

Run `python elect.py` and compare it against the existing methods side by
side. See `research.md` for candidate methods worth implementing.

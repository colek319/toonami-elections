"""Tests for the election pipeline. Run with: pytest"""

import numpy as np
import pandas as pd
import pytest

from constants import DIMENSIONS, NOMINATIONS
from elect import Election, profiles
from election_strategies import euclidean, mahalanobis
from imputation import impute_column_mean, impute_voter_bias
from loaders import Load


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_csv(tmp_path):
    """A tiny Google Forms-style export with the real export's quirks:

    junk columns (Timestamp, Email Address, "[Row 7]"), a double-space show
    name, one missing cell (Bob skipped Alpha's Aesthetic), and one show
    ("Ghost") nobody rated at all.
    """
    shows = ["Alpha", "Beta  Two", "Ghost"]
    header = ["Timestamp", "Name"]
    for show in shows:
        header += [f"{show} [{dim}]" for dim in DIMENSIONS]
    header += ["Beta  Two [Row 7]", "Email Address"]

    ann = ["1/1/2026 0:00", "Ann"] + ["1"] * 6 + ["5"] * 6 + [""] * 6 + ["", "ann@x.com"]
    bob = ["1/1/2026 0:01", "Bob"] + ["3"] * 5 + [""] + ["4"] * 6 + [""] * 6 + ["7", "bob@x.com"]

    path = tmp_path / "mini.csv"
    lines = [",".join(header), ",".join(ann), ",".join(bob)]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def ratings(mini_csv):
    return Load(mini_csv)()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_load_shape_and_shows(ratings):
    assert list(ratings.index) == ["Ann", "Bob"]
    # "Beta  Two" is whitespace-normalized; unrated "Ghost" is dropped.
    assert set(ratings.columns.get_level_values("show")) == {"Alpha", "Beta Two"}
    assert set(ratings.columns.get_level_values("dim")) == set(DIMENSIONS)


def test_load_keeps_missing_votes_as_nan(ratings):
    assert np.isnan(ratings.loc["Bob", ("Alpha", "Aesthetic")])
    assert ratings.loc["Ann", ("Alpha", "Aesthetic")] == 1.0


def test_load_drops_junk_columns(ratings):
    # Nothing from Timestamp / Email Address / "[Row 7]" survives parsing:
    # every column is a (show, dimension) pair with a real dimension.
    assert all(dim in DIMENSIONS for dim in ratings.columns.get_level_values("dim"))


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("imputer", [impute_column_mean, impute_voter_bias])
def test_imputers_fill_everything_and_keep_observed(imputer, ratings):
    out = imputer(ratings)
    assert not out.isna().any().any()
    observed = ratings.notna()
    assert out[observed].equals(ratings[observed])


@pytest.mark.parametrize("imputer", [impute_column_mean, impute_voter_bias])
def test_imputers_do_not_mutate_input(imputer, ratings):
    before = ratings.copy()
    imputer(ratings)
    assert ratings.equals(before)


def test_column_mean_fills_with_crowd_mean(ratings):
    out = impute_column_mean(ratings)
    # Only Ann rated Alpha's Aesthetic (a 1), so Bob's gap becomes 1.0.
    assert out.loc["Bob", ("Alpha", "Aesthetic")] == 1.0


def test_voter_bias_shifts_fill_by_rater_generosity(ratings):
    out = impute_voter_bias(ratings)
    col_mean = 1.0  # Ann's lone Aesthetic rating for Alpha
    bias = ratings.loc["Bob"].mean() - np.nanmean(ratings.to_numpy())
    assert out.loc["Bob", ("Alpha", "Aesthetic")] == pytest.approx(col_mean + bias)
    assert ((out >= 1) & (out <= 5)).all().all()


# ---------------------------------------------------------------------------
# Election strategies
# ---------------------------------------------------------------------------

def make_profiles(rows):
    return pd.DataFrame.from_dict(rows, orient="index", dtype=float).set_axis(
        DIMENSIONS, axis=1
    )


def test_euclidean_distance_from_centroid():
    past = make_profiles({"P1": [2] * 6, "P2": [4] * 6})  # centroid: all 3s
    noms = make_profiles({"AtCentroid": [3] * 6, "OffByThree": [6, 3, 3, 3, 3, 3]})
    d = euclidean(past, noms)
    assert d["AtCentroid"] == pytest.approx(0.0)
    assert d["OffByThree"] == pytest.approx(3.0)


def test_mahalanobis_penalizes_low_variance_dimensions():
    # Past shows vary a lot on Goon, barely on Cute. An equal-sized miss
    # should cost more on Cute (the club is consistent there) than on Goon.
    c = [3.0] * 6
    past = make_profiles({
        "P1": [5, 3, 3, 3, 3, 3],
        "P2": [1, 3, 3, 3, 3, 3],
        "P3": [3, 3.2, 3, 3, 3, 3],
        "P4": [3, 2.8, 3, 3, 3, 3],
    })
    noms = make_profiles({
        "MissOnGoon": [4, 3, 3, 3, 3, 3],
        "MissOnCute": [3, 4, 3, 3, 3, 3],
    })
    d = mahalanobis(past, noms)
    assert d["MissOnCute"] > d["MissOnGoon"]
    # Euclidean can't tell them apart — that's the point of the strategy.
    e = euclidean(past, noms)
    assert e["MissOnGoon"] == pytest.approx(e["MissOnCute"])


# ---------------------------------------------------------------------------
# Election
# ---------------------------------------------------------------------------

def test_election_end_to_end(mini_csv):
    election = Election(impute_column_mean, Load(mini_csv), ["Beta Two"], euclidean, n=1)
    winners = election.run()
    assert list(winners.index) == ["Beta Two"]
    assert winners["Beta Two"] >= 0


def test_election_ranking_is_sorted_ascending(mini_csv):
    election = Election(impute_column_mean, Load(mini_csv), ["Beta Two"], euclidean)
    assert election.ranking().is_monotonic_increasing


def test_profiles_are_show_by_dimension(ratings):
    prof = profiles(impute_column_mean(ratings))
    assert list(prof.columns) == DIMENSIONS
    assert set(prof.index) == {"Alpha", "Beta Two"}
    assert ((prof >= 1) & (prof <= 5)).all().all()


def test_real_csv_smoke():
    """The season's actual election runs and elects 3 of the nominations."""
    election = Election(impute_column_mean, Load(), NOMINATIONS, euclidean)
    winners = election.run()
    assert len(winners) == 3
    assert set(winners.index) <= set(NOMINATIONS)
    assert election.ranking().is_monotonic_increasing

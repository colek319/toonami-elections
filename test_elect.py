"""Tests for the election pipeline. Run with: pytest"""

import numpy as np
import pandas as pd
import pytest

from constants import DIMENSIONS, SUMMER_2026_NOMINATIONS
from elect import Election, profiles
from election_strategies import euclidean, mahalanobis, portfolio
from imputation import impute_column_mean, impute_knn, impute_voter_bias
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

@pytest.mark.parametrize("imputer", [impute_column_mean, impute_knn, impute_voter_bias])
def test_imputers_fill_everything_and_keep_observed(imputer, ratings):
    out = imputer(ratings)
    assert not out.isna().any().any()
    observed = ratings.notna()
    assert out[observed].equals(ratings[observed])


@pytest.mark.parametrize("imputer", [impute_column_mean, impute_knn, impute_voter_bias])
def test_imputers_do_not_mutate_input(imputer, ratings):
    before = ratings.copy()
    imputer(ratings)
    assert ratings.equals(before)


def test_column_mean_fills_with_crowd_mean(ratings):
    out = impute_column_mean(ratings)
    # Only Ann rated Alpha's Aesthetic (a 1), so Bob's gap becomes 1.0.
    assert out.loc["Bob", ("Alpha", "Aesthetic")] == 1.0


def test_knn_fills_from_correlated_neighbour(ratings):
    # Gap's ballot tracks Twin's exactly and mirrors Foil's. The missing
    # cell should come from Twin (a 5), not the anti-correlated Foil and
    # not the crowd mean (which Foil's 1 would drag to 3).
    cols = pd.MultiIndex.from_product(
        [["S1", "S2"], DIMENSIONS], names=["show", "dim"]
    )
    df = pd.DataFrame.from_dict(
        {
            "Gap": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 4, np.nan],
            "Twin": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 4, 5],
            "Foil": [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 2, 1],
        },
        orient="index",
        dtype=float,
    ).set_axis(cols, axis=1)
    out = impute_knn(df)
    assert out.loc["Gap", ("S2", "Aesthetic")] == 5.0


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


def test_portfolio_elects_a_complementary_slate():
    # Centroid: all 3s. Hot and Cold each miss it by 2 on Goon but in
    # opposite directions, so {Hot, Cold, Mid} averages exactly onto the
    # centroid. Euclidean would take the Mehs over Hot and Cold.
    past = make_profiles({"P1": [2] * 6, "P2": [4] * 6})
    noms = make_profiles({
        "Hot": [5, 3, 3, 3, 3, 3],
        "Cold": [1, 3, 3, 3, 3, 3],
        "Mid": [3, 3, 3, 3, 3, 3],
        "Meh1": [3.6, 3, 3, 3, 3, 3],
        "Meh2": [3.6, 3, 3, 3, 3, 3],
    })
    d = portfolio(past, noms)
    for show in ["Hot", "Cold", "Mid"]:
        assert d[show] == pytest.approx(0.0)
    assert d["Meh1"] > 0 and d["Meh2"] > 0
    e = euclidean(past, noms)
    assert e["Meh1"] < e["Hot"]  # the individual view disagrees


# ---------------------------------------------------------------------------
# Election
# ---------------------------------------------------------------------------

def test_election_end_to_end(mini_csv):
    election = Election(impute_column_mean, Load(mini_csv), ["Beta Two"], euclidean, n=1)
    winners = election.run()
    assert list(winners.index) == ["Beta Two"]
    assert winners["Beta Two"] >= 0


def test_election_rejects_unknown_nomination(mini_csv):
    # A typo'd or renamed show fails fast with the bad names, not a
    # cryptic pandas KeyError deep in the strategy.
    election = Election(
        impute_column_mean, Load(mini_csv), ["Beta Two", "Nonexistent"], euclidean
    )
    with pytest.raises(ValueError, match="Nonexistent"):
        election.ranking()


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
    election = Election(impute_column_mean, Load(), SUMMER_2026_NOMINATIONS, euclidean)
    winners = election.run()
    assert len(winners) == 3
    assert set(winners.index) <= set(SUMMER_2026_NOMINATIONS)
    assert election.ranking().is_monotonic_increasing

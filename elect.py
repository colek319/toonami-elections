"""Toonami vote wizard election script.

Watchers scored every show from Toonami past on six dimensions
(Goon, Cute, Laugh, Edgy, Rad, Aesthetic), 1-5. The shows already
voted in over past seasons define the club's "taste centroid" — an
average point in 6-D dimension space. The election picks the 3
nominations whose profiles land closest to that point.

MINIMIZE FREE ENERGY: the winning shows are the least surprising
given what the club has historically watched.
"""

from constants import DIMENSIONS, SUMMER_2026_NOMINATIONS
from election_strategies import euclidean, mahalanobis, portfolio
from imputation import impute_column_mean, impute_knn, impute_voter_bias
from loaders import Load


def profiles(df):
    """Each show's mean score per dimension -> show x dimension DataFrame."""
    return df.mean(axis=0).unstack("dim")[DIMENSIONS]


class Election:
    """One season's election: how to impute, what to load, who's running, how to score.

    >>> Election(impute_voter_bias, Load("data/toonami-jul-2026.csv"), SUMMER_2026_NOMINATIONS, mahalanobis).run()
    """

    def __init__(self, imputer, loader, targets, strategy, n=3):
        self.imputer = imputer
        self.loader = loader
        self.targets = list(targets)
        self.strategy = strategy
        self.n = n

    def ranking(self):
        """Distance for every target, lowest (least surprising) first."""
        prof = profiles(self.imputer(self.loader()))
        missing = [t for t in self.targets if t not in prof.index]
        if missing:
            raise ValueError(
                f"nominations not in the ballot data: {missing} — "
                "check they match the CSV column names exactly"
            )
        past = prof.drop(index=self.targets)
        noms = prof.loc[self.targets]
        return self.strategy(past, noms).sort_values()

    def run(self):
        """The n winners as a Series of (show -> distance)."""
        return self.ranking().head(self.n)


def main():
    elections = {
        "column-mean impute + Euclidean": Election(
            impute_column_mean, Load("data/toonami-jul-2026.csv"), SUMMER_2026_NOMINATIONS, euclidean
        ),
        "voter-bias impute + Mahalanobis": Election(
            impute_voter_bias, Load("data/toonami-jul-2026.csv"), SUMMER_2026_NOMINATIONS, mahalanobis
        ),
        "knn impute + portfolio": Election(
            impute_knn, Load("data/toonami-jul-2026.csv"), SUMMER_2026_NOMINATIONS, portfolio
        ),
    }

    for name, election in elections.items():
        print(f"=== {name} ===")
        print(election.ranking().round(3).to_string())
        print("winners:", ", ".join(election.run().index), "\n")


if __name__ == "__main__":
    main()

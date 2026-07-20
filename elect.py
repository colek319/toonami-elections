"""Toonami July 2026 — vote wizard election script.

Watchers scored every show from Toonami past on six dimensions
(Goon, Cute, Laugh, Edgy, Rad, Aesthetic), 1-5. The shows already
voted in over past seasons define the club's "taste centroid" — an
average point in 6-D dimension space. The election picks the 3
nominations whose profiles land closest to that point.

MINIMIZE FREE ENERGY: the winning shows are the least surprising
given what the club has historically watched.
"""

from constants import DIMENSIONS, NOMINATIONS
from election_strategies import euclidean, mahalanobis
from imputation import impute_column_mean, impute_voter_bias
from loaders import Load


def profiles(df):
    """Each show's mean score per dimension -> show x dimension DataFrame."""
    return df.mean(axis=0).unstack("dim")[DIMENSIONS]


class Election:
    """One season's election: how to impute, what to load, who's running, how to score.

    >>> Election(impute_voter_bias, Load("toonami-jul-2026.csv"), NOMINATIONS, mahalanobis).run()
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
        past = prof.drop(index=self.targets)
        noms = prof.loc[self.targets]
        return self.strategy(past, noms).sort_values()

    def run(self):
        """The n winners as a Series of (show -> distance)."""
        return self.ranking().head(self.n)


def main():
    elections = {
        "column-mean impute + Euclidean": Election(
            impute_column_mean, Load("toonami-jul-2026.csv"), NOMINATIONS, euclidean
        ),
        "voter-bias impute + Mahalanobis": Election(
            impute_voter_bias, Load("toonami-jul-2026.csv"), NOMINATIONS, mahalanobis
        ),
    }

    for name, election in elections.items():
        print(f"=== {name} ===")
        print(election.ranking().round(3).to_string())
        print("winners:", ", ".join(election.run().index), "\n")


if __name__ == "__main__":
    main()

"""Season constants for the Toonami elections."""

import re
from pathlib import Path

CSV_PATH = Path(__file__).parent / "data/toonami-jul-2026.csv"

DIMENSIONS = ["Goon", "Cute", "Laugh", "Edgy", "Rad", "Aesthetic"]

# Each season's nominations are frozen under a season-specific name so the
# README's past-elections table stays reproducible (names as they appear in
# that season's CSV).
SUMMER_2026_NOMINATIONS = [
    "Jaadugar: A Witch In Mongolia",
    "The 100 Girlfriends Who Really Really Really Really REALLY Love You season 3",
    "You and I Are Polar Opposites season 2",
    "Sparks of Tomorrow",
    "Smoking Behind the Supermarket With You",
    "Chainsmoker Cat",
    "Let's Go Kaikigumi",
]

COLUMN_RE = re.compile(r"^(?P<show>.+) \[(?P<dim>" + "|".join(DIMENSIONS) + r")\]$")

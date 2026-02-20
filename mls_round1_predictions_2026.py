#!/usr/bin/env python3
"""
MLS 2026 Round 1 Match Predictions
====================================
Predicts outcomes for MLS Opening Weekend (February 21-22, 2026)
using an Elo rating system calibrated on 2022-2025 MLS seasons.

Methodology
-----------
1. Elo ratings are computed from each team's final points tally
   across the 2022-2025 regular seasons.
2. Recent seasons are weighted more heavily (exponential decay).
3. A home-field advantage of +100 Elo points is applied.
4. Three-way win/draw/loss probabilities are derived from the
   Elo differential, with MLS-calibrated draw rate (~22.5%).
5. Off-season roster changes are approximated by a 30% regression
   toward the league mean (1500) each year.

Data Sources
------------
- 2025 MLS Final Standings: Inter Miami CF won MLS Cup (3-1 vs Vancouver).
  Philadelphia Union won the Supporters' Shield with 66 pts.
- 2024 MLS Final Standings: LA Galaxy won MLS Cup; Inter Miami won Shield.
- 2023 MLS Final Standings: Columbus Crew won MLS Cup.
- 2022 MLS Final Standings: LAFC won MLS Cup.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  Model Configuration
# ─────────────────────────────────────────────────────────────

INITIAL_ELO: float = 1500.0
HOME_ADVANTAGE: float = 100.0   # Elo points added to home team
K_SEASON: float = 40.0          # K-factor for season-level Elo update
REGRESSION_TO_MEAN: float = 0.30  # Off-season mean-reversion factor
DRAW_BASE_RATE: float = 0.225    # MLS historical draw rate ~22-23%
DRAW_SENSITIVITY: float = 0.12   # Extra draw probability when sides are even
ELO_SCALE: float = 400.0         # Standard Elo scale constant

# Season weights: more recent seasons matter more
SEASON_WEIGHTS: Dict[int, float] = {
    2022: 0.10,
    2023: 0.15,
    2024: 0.25,
    2025: 0.50,
}

MLS_AVG_POINTS: float = 47.5     # Historical league-average points per 34-game season
POINTS_TO_ELO_SCALE: float = 4.5  # 10 pts above avg ≈ +45 Elo


# ─────────────────────────────────────────────────────────────
#  Historical Season Data (points, games_played)
# ─────────────────────────────────────────────────────────────

HISTORICAL_SEASONS: Dict[int, Dict[str, Tuple[int, int]]] = {
    2022: {
        # Eastern Conference
        "Philadelphia Union":     (71, 34),
        "New York City FC":       (62, 34),
        "New York Red Bulls":     (59, 34),
        "CF Montréal":            (58, 34),
        "Orlando City SC":        (54, 34),
        "Nashville SC":           (53, 34),
        "Columbus Crew":          (52, 34),
        "Atlanta United FC":      (51, 34),
        "Charlotte FC":           (50, 34),
        "New England Revolution": (47, 34),
        "D.C. United":            (45, 34),
        "Chicago Fire FC":        (45, 34),
        "FC Cincinnati":          (37, 34),
        "Toronto FC":             (33, 34),
        "Inter Miami CF":         (27, 34),
        # Western Conference
        "LAFC":                   (77, 34),  # MLS Cup winners
        "Austin FC":              (62, 34),
        "FC Dallas":              (59, 34),
        "Sporting Kansas City":   (58, 34),
        "Houston Dynamo FC":      (57, 34),
        "Colorado Rapids":        (55, 34),
        "LA Galaxy":              (51, 34),
        "Real Salt Lake":         (50, 34),
        "Seattle Sounders FC":    (47, 34),
        "Portland Timbers":       (47, 34),
        "Minnesota United FC":    (44, 34),
        "Vancouver Whitecaps FC": (40, 34),
        "San Jose Earthquakes":   (36, 34),
        # St. Louis City SC debuted in 2023
    },
    2023: {
        # Eastern Conference
        "FC Cincinnati":          (73, 34),  # Supporters' Shield
        "Philadelphia Union":     (65, 34),
        "Inter Miami CF":         (58, 34),  # Messi arrived mid-season
        "Orlando City SC":        (57, 34),
        "Nashville SC":           (56, 34),
        "New York Red Bulls":     (52, 34),
        "Columbus Crew":          (50, 34),  # MLS Cup winners
        "Charlotte FC":           (50, 34),
        "New York City FC":       (49, 34),
        "Chicago Fire FC":        (48, 34),
        "Atlanta United FC":      (46, 34),
        "New England Revolution": (43, 34),
        "D.C. United":            (39, 34),
        "Toronto FC":             (36, 34),
        "CF Montréal":            (32, 34),
        # Western Conference
        "FC Dallas":              (63, 34),
        "LA Galaxy":              (60, 34),
        "LAFC":                   (60, 34),
        "Real Salt Lake":         (58, 34),
        "Seattle Sounders FC":    (57, 34),
        "Houston Dynamo FC":      (54, 34),
        "Portland Timbers":       (51, 34),
        "Colorado Rapids":        (51, 34),
        "Austin FC":              (50, 34),
        "Sporting Kansas City":   (49, 34),
        "Vancouver Whitecaps FC": (46, 34),
        "Minnesota United FC":    (44, 34),
        "San Jose Earthquakes":   (43, 34),
        "St. Louis City SC":      (39, 34),  # Debut season
    },
    2024: {
        # Eastern Conference
        "Inter Miami CF":         (74, 34),  # Supporters' Shield
        "Philadelphia Union":     (65, 34),
        "FC Cincinnati":          (62, 34),
        "New York City FC":       (55, 34),
        "Columbus Crew":          (54, 34),
        "Nashville SC":           (53, 34),
        "Charlotte FC":           (52, 34),
        "Orlando City SC":        (48, 34),
        "Atlanta United FC":      (47, 34),
        "Chicago Fire FC":        (45, 34),
        "New York Red Bulls":     (44, 34),
        "CF Montréal":            (40, 34),
        "D.C. United":            (38, 34),
        "New England Revolution": (37, 34),
        "Toronto FC":             (28, 34),
        # Western Conference
        "LA Galaxy":              (63, 34),  # MLS Cup winners
        "LAFC":                   (63, 34),
        "Real Salt Lake":         (60, 34),
        "Colorado Rapids":        (57, 34),
        "Minnesota United FC":    (54, 34),
        "Seattle Sounders FC":    (53, 34),
        "Vancouver Whitecaps FC": (50, 34),
        "Austin FC":              (47, 34),
        "Portland Timbers":       (46, 34),
        "FC Dallas":              (45, 34),
        "St. Louis City SC":      (44, 34),
        "Houston Dynamo FC":      (43, 34),
        "Sporting Kansas City":   (39, 34),
        "San Jose Earthquakes":   (37, 34),
        # San Diego FC debuted in 2025
    },
    2025: {
        # Eastern Conference
        "Philadelphia Union":     (66, 34),  # Supporters' Shield
        "FC Cincinnati":          (65, 34),
        "Inter Miami CF":         (65, 34),  # MLS Cup winners
        "Charlotte FC":           (59, 34),
        "New York City FC":       (56, 34),
        "Nashville SC":           (54, 34),
        "Columbus Crew":          (54, 34),
        "Chicago Fire FC":        (53, 34),
        "Orlando City SC":        (53, 34),
        "New York Red Bulls":     (43, 34),
        "New England Revolution": (36, 34),
        "Toronto FC":             (32, 34),
        "CF Montréal":            (28, 34),
        "Atlanta United FC":      (28, 34),
        "D.C. United":            (26, 34),
        # Western Conference
        "San Diego FC":           (70, 34),  # Conference leaders, debut season
        "Vancouver Whitecaps FC": (63, 34),  # MLS Cup finalists
        "LAFC":                   (60, 34),
        "Minnesota United FC":    (58, 34),
        "Seattle Sounders FC":    (55, 34),
        "Austin FC":              (52, 34),
        "LA Galaxy":              (50, 34),
        "FC Dallas":              (44, 34),
        "Portland Timbers":       (44, 34),
        "Real Salt Lake":         (41, 34),
        "San Jose Earthquakes":   (41, 34),
        "Colorado Rapids":        (41, 34),
        "St. Louis City SC":      (35, 34),
        "Sporting Kansas City":   (33, 34),
        "Houston Dynamo FC":      (31, 34),
    },
}


# ─────────────────────────────────────────────────────────────
#  Round 1 Fixtures (February 21-22, 2026)
# ─────────────────────────────────────────────────────────────

@dataclass
class Fixture:
    home: str
    away: str
    kickoff_et: str
    date: str
    broadcast: str = "Apple TV"

ROUND_1_FIXTURES: List[Fixture] = [
    # Saturday, February 21 ─────────────────────────────────
    Fixture("St. Louis City SC",      "Charlotte FC",           "2:30 PM",  "Feb 21", "Apple TV"),
    Fixture("FC Cincinnati",          "Atlanta United FC",      "4:30 PM",  "Feb 21", "FOX / Apple TV"),
    Fixture("Orlando City SC",        "New York Red Bulls",     "7:30 PM",  "Feb 21", "Apple TV"),
    Fixture("D.C. United",            "Minnesota United FC",    "7:30 PM",  "Feb 21", "Apple TV"),
    Fixture("Vancouver Whitecaps FC", "Real Salt Lake",         "7:30 PM",  "Feb 21", "Apple TV"),
    Fixture("Nashville SC",           "New England Revolution", "8:30 PM",  "Feb 21", "Apple TV"),
    Fixture("Austin FC",              "Philadelphia Union",     "8:30 PM",  "Feb 21", "Apple TV"),
    Fixture("Houston Dynamo FC",      "Chicago Fire FC",        "8:30 PM",  "Feb 21", "Apple TV"),
    Fixture("FC Dallas",              "Toronto FC",             "8:30 PM",  "Feb 21", "Apple TV"),
    Fixture("LAFC",                   "Inter Miami CF",         "9:30 PM",  "Feb 21", "Apple TV"),
    Fixture("Portland Timbers",       "Columbus Crew",          "10:30 PM", "Feb 21", "Apple TV"),
    Fixture("San Jose Earthquakes",   "Sporting Kansas City",   "10:30 PM", "Feb 21", "Apple TV"),
    Fixture("San Diego FC",           "CF Montréal",            "10:30 PM", "Feb 21", "Apple TV"),
    # Sunday, February 22 ────────────────────────────────────
    Fixture("LA Galaxy",              "New York City FC",       "7:00 PM",  "Feb 22", "Apple TV"),
    Fixture("Seattle Sounders FC",    "Colorado Rapids",        "9:15 PM",  "Feb 22", "Apple TV"),
]


# ─────────────────────────────────────────────────────────────
#  Elo Engine
# ─────────────────────────────────────────────────────────────

def season_elo_from_points(pts: int, gp: int, league_avg: float = MLS_AVG_POINTS) -> float:
    """
    Convert a team's final season points into a season-level Elo score.

    We treat league-average performance as Elo 1500 and scale linearly:
        season_elo = 1500 + (pts_per_game - avg_ppg) * pts_to_elo_scale * 34

    The '34' normalises for any abbreviated seasons.
    """
    pts_per_game = pts / gp
    avg_ppg = league_avg / 34.0
    delta = (pts_per_game - avg_ppg) * POINTS_TO_ELO_SCALE * 34.0
    return INITIAL_ELO + delta


def build_elo_ratings() -> Dict[str, float]:
    """
    Compute 2026 pre-season Elo ratings for every MLS team.

    Algorithm
    ---------
    1. For every season, convert each team's points to a 'season Elo'.
    2. Compute a weighted average across seasons (recent = more weight).
    3. Apply 30% regression toward 1500 to simulate off-season uncertainty.
    4. Teams new to the league start at 1480 (slight discount for unknowns)
       and gain full exposure from their first recorded season onward.
    """
    all_teams: set = set()
    for season_data in HISTORICAL_SEASONS.values():
        all_teams.update(season_data.keys())

    # Collect weighted Elo per team
    weighted_elo: Dict[str, List[Tuple[float, float]]] = {t: [] for t in all_teams}
    for season, season_data in HISTORICAL_SEASONS.items():
        weight = SEASON_WEIGHTS[season]
        for team, (pts, gp) in season_data.items():
            elo = season_elo_from_points(pts, gp)
            weighted_elo[team].append((elo, weight))

    ratings: Dict[str, float] = {}
    for team, elo_weight_pairs in weighted_elo.items():
        total_weight = sum(w for _, w in elo_weight_pairs)
        raw_elo = sum(e * w for e, w in elo_weight_pairs) / total_weight
        # Regression to mean
        regressed = INITIAL_ELO + (raw_elo - INITIAL_ELO) * (1.0 - REGRESSION_TO_MEAN)
        ratings[team] = round(regressed, 1)

    return ratings


def expected_score(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected score for player A versus player B."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))


def predict_three_way(
    home_elo: float,
    away_elo: float,
    home_adv: float = HOME_ADVANTAGE,
    draw_base: float = DRAW_BASE_RATE,
    draw_sensitivity: float = DRAW_SENSITIVITY,
) -> Tuple[float, float, float]:
    """
    Predict home-win / draw / away-win probabilities from Elo ratings.

    Steps
    -----
    1. Apply home-field advantage to the home team's rating.
    2. Compute the raw home-win probability using the Elo formula.
    3. Estimate draw probability: draws are more likely when teams are
       evenly matched (raw win probability close to 50 %).
    4. Scale win/loss probabilities into the remaining probability mass.

    Returns
    -------
    (p_home_win, p_draw, p_away_win)  — sums to 1.0
    """
    adjusted_home = home_elo + home_adv
    raw_home_win = expected_score(adjusted_home, away_elo)

    # Draw is highest when raw_home_win ≈ 0.5
    evenness = 1.0 - abs(raw_home_win - 0.5) * 2.0   # 0 (one-sided) to 1 (even)
    p_draw = draw_base + draw_sensitivity * evenness

    remaining = 1.0 - p_draw
    p_home_win = raw_home_win * remaining
    p_away_win = (1.0 - raw_home_win) * remaining

    return p_home_win, p_draw, p_away_win


def confidence_label(prob: float) -> str:
    """Human-readable confidence tier for a win probability."""
    if prob >= 0.60:
        return "HIGH"
    if prob >= 0.45:
        return "MODERATE"
    if prob >= 0.35:
        return "LOW"
    return "TOSS-UP"


# ─────────────────────────────────────────────────────────────
#  Prediction Report
# ─────────────────────────────────────────────────────────────

@dataclass
class MatchPrediction:
    fixture: Fixture
    home_elo: float
    away_elo: float
    p_home: float
    p_draw: float
    p_away: float

    @property
    def predicted_outcome(self) -> str:
        probs = {"Home Win": self.p_home, "Draw": self.p_draw, "Away Win": self.p_away}
        return max(probs, key=probs.__getitem__)

    @property
    def predicted_winner(self) -> str:
        outcome = self.predicted_outcome
        if outcome == "Home Win":
            return self.fixture.home
        if outcome == "Away Win":
            return self.fixture.away
        return "Draw"

    @property
    def confidence(self) -> str:
        max_p = max(self.p_home, self.p_draw, self.p_away)
        return confidence_label(max_p)


def run_predictions() -> List[MatchPrediction]:
    ratings = build_elo_ratings()
    predictions: List[MatchPrediction] = []

    for fixture in ROUND_1_FIXTURES:
        home_elo = ratings.get(fixture.home, INITIAL_ELO)
        away_elo = ratings.get(fixture.away, INITIAL_ELO)
        p_home, p_draw, p_away = predict_three_way(home_elo, away_elo)
        predictions.append(
            MatchPrediction(
                fixture=fixture,
                home_elo=home_elo,
                away_elo=away_elo,
                p_home=p_home,
                p_draw=p_draw,
                p_away=p_away,
            )
        )
    return predictions


def print_report(predictions: List[MatchPrediction]) -> None:
    header = "=" * 78
    divider = "-" * 78

    print(header)
    print("  MLS 2026 ROUND 1 — MATCH PREDICTIONS (Opening Weekend)")
    print("  Model: Elo Ratings | Seasons: 2022-2025 | Updated: Feb 2026")
    print(header)

    current_date = ""
    for pred in predictions:
        if pred.fixture.date != current_date:
            current_date = pred.fixture.date
            day_label = "SATURDAY, FEBRUARY 21" if "21" in current_date else "SUNDAY, FEBRUARY 22"
            print(f"\n  {day_label}, 2026")
            print(divider)

        home = pred.fixture.home
        away = pred.fixture.away
        kickoff = pred.fixture.kickoff_et
        broadcast = pred.fixture.broadcast

        elo_diff = pred.home_elo - pred.away_elo
        elo_diff_str = f"+{elo_diff:.0f}" if elo_diff >= 0 else f"{elo_diff:.0f}"

        print(f"\n  {kickoff} ET  |  {broadcast}")
        print(f"  {home} vs {away}")
        print(f"  Elo: {pred.home_elo:.0f}  vs  {pred.away_elo:.0f}  (home advantage net: {elo_diff_str})")
        print(
            f"  Probabilities:  "
            f"Home {pred.p_home:.1%}  |  "
            f"Draw {pred.p_draw:.1%}  |  "
            f"Away {pred.p_away:.1%}"
        )
        print(
            f"  Prediction: {pred.predicted_winner:<30s}  "
            f"[{pred.confidence} confidence]"
        )

    print(f"\n{divider}")
    print("  SUMMARY — PREDICTED WINNERS")
    print(divider)
    print(f"  {'Match':<46} {'Outcome':<14} {'Prob':>6}")
    print(f"  {'-'*46} {'-'*14} {'-'*6}")
    for pred in predictions:
        matchup = f"{pred.fixture.home} vs {pred.fixture.away}"
        if len(matchup) > 44:
            matchup = matchup[:44]
        outcome = pred.predicted_winner
        if outcome == "Draw":
            best_prob = pred.p_draw
        elif outcome == pred.fixture.home:
            best_prob = pred.p_home
        else:
            best_prob = pred.p_away
        print(f"  {matchup:<46} {outcome:<14} {best_prob:>5.1%}")

    print(f"\n{divider}")
    print("  TEAM ELO RATINGS — 2026 PRE-SEASON")
    print(divider)

    all_teams = {pred.fixture.home for pred in predictions} | {pred.fixture.away for pred in predictions}
    ratings_subset = {}
    full_ratings = build_elo_ratings()
    for team in all_teams:
        ratings_subset[team] = full_ratings.get(team, INITIAL_ELO)
    sorted_ratings = sorted(ratings_subset.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, elo) in enumerate(sorted_ratings, 1):
        bar_width = int((elo - 1350) / 10)
        bar = "#" * bar_width
        print(f"  {rank:>2}. {team:<30s}  {elo:>6.0f}  {bar}")

    print(f"\n{header}")
    print("  DISCLAIMER")
    print(
        "  Predictions are probabilistic estimates based on team performance\n"
        "  from 2022-2025 regular seasons. They do not account for individual\n"
        "  match day form, injuries, roster moves, or weather conditions.\n"
        "  Use alongside qualitative analysis for best results."
    )
    print(header)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preds = run_predictions()
    print_report(preds)

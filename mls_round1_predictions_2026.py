#!/usr/bin/env python3
"""
MLS 2026 Round 1 Match Predictions
====================================
Predicts outcomes AND Over/Under 2.5 goals for MLS Opening Weekend
(February 21-22, 2026) using two complementary models.

Models
------
1. **Elo Model** — predicts home-win / draw / away-win probabilities.
   Calibrated from 2022-2025 final regular-season points.

2. **Poisson Goal Model** — predicts expected goals and P(Over 2.5).
   Uses each team's 2025 goals-for / goals-against per game as
   attack / defense strength ratings, then applies a Dixon-Coles style
   independent Poisson approximation to get total-goal distributions.

Both models apply a regression-to-mean to account for off-season
roster changes and the general uncertainty at season start.

Data Sources
------------
- 2025 MLS Final Standings (GF/GA, all 30 teams):
    Eastern: Philadelphia Union 57/35, FC Cincinnati 52/40,
    Inter Miami 81/55, Charlotte 55/46, NYCFC 50/44, Nashville 58/45,
    Columbus 55/51, Chicago Fire 68/60, Orlando 63/51, NYRB 48/47,
    NE Rev 44/51, Toronto 37/44, Montréal 34/60, Atlanta 38/63, DC 30/66.
    Western: San Diego 64/41, Vancouver 66/38, LAFC 65/40,
    Minnesota 56/39, Seattle 58/48, Austin 37/45, FC Dallas 52/55,
    Portland 41/48, RSL 38/49, San Jose 60/63, Colorado 44/56,
    Houston 43/56, St. Louis 44/58, LA Galaxy 46/66, Sporting KC 46/70.
- 2025 MLS Cup: Inter Miami 3-1 Vancouver; Supporters' Shield: Philadelphia Union
- 2024 MLS Cup: LA Galaxy 2-1 NYRB; Supporters' Shield: Inter Miami (74 pts)
- 2023 MLS Cup: Columbus Crew; 2022 MLS Cup: LAFC
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  Shared Configuration
# ─────────────────────────────────────────────────────────────

INITIAL_ELO: float = 1500.0
HOME_ADVANTAGE: float = 100.0    # Elo points added to home team
K_SEASON: float = 40.0
REGRESSION_TO_MEAN: float = 0.30
DRAW_BASE_RATE: float = 0.225
DRAW_SENSITIVITY: float = 0.12
ELO_SCALE: float = 400.0

SEASON_WEIGHTS: Dict[int, float] = {2022: 0.10, 2023: 0.15, 2024: 0.25, 2025: 0.50}

MLS_AVG_POINTS: float = 47.5
POINTS_TO_ELO_SCALE: float = 4.5

# Poisson / goal model parameters
LEAGUE_HOME_AVG: float = 1.60   # MLS home team avg goals/game
LEAGUE_AWAY_AVG: float = 1.30   # MLS away team avg goals/game
LEAGUE_AVG_GPG: float = 1.45    # Per-team average (total/2 teams per game)
GOAL_REGRESSION: float = 0.25   # Off-season regression for attack/defense


# ─────────────────────────────────────────────────────────────
#  Historical Season Points (for Elo model)
# ─────────────────────────────────────────────────────────────

HISTORICAL_SEASONS: Dict[int, Dict[str, Tuple[int, int]]] = {
    2022: {
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
        "LAFC":                   (77, 34),
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
    },
    2023: {
        "FC Cincinnati":          (73, 34),
        "Philadelphia Union":     (65, 34),
        "Inter Miami CF":         (58, 34),
        "Orlando City SC":        (57, 34),
        "Nashville SC":           (56, 34),
        "New York Red Bulls":     (52, 34),
        "Columbus Crew":          (50, 34),
        "Charlotte FC":           (50, 34),
        "New York City FC":       (49, 34),
        "Chicago Fire FC":        (48, 34),
        "Atlanta United FC":      (46, 34),
        "New England Revolution": (43, 34),
        "D.C. United":            (39, 34),
        "Toronto FC":             (36, 34),
        "CF Montréal":            (32, 34),
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
        "St. Louis City SC":      (39, 34),
    },
    2024: {
        "Inter Miami CF":         (74, 34),
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
        "LA Galaxy":              (64, 34),
        "LAFC":                   (64, 34),
        "Real Salt Lake":         (59, 34),
        "Colorado Rapids":        (50, 34),
        "Minnesota United FC":    (52, 34),
        "Seattle Sounders FC":    (53, 34),
        "Vancouver Whitecaps FC": (47, 34),
        "Austin FC":              (42, 34),
        "Portland Timbers":       (46, 34),
        "FC Dallas":              (45, 34),
        "St. Louis City SC":      (44, 34),
        "Houston Dynamo FC":      (43, 34),
        "Sporting Kansas City":   (39, 34),
        "San Jose Earthquakes":   (37, 34),
    },
    2025: {
        "Philadelphia Union":     (66, 34),
        "FC Cincinnati":          (65, 34),
        "Inter Miami CF":         (65, 34),
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
        "San Diego FC":           (63, 34),
        "Vancouver Whitecaps FC": (63, 34),
        "LAFC":                   (60, 34),
        "Minnesota United FC":    (58, 34),
        "Seattle Sounders FC":    (55, 34),
        "Austin FC":              (47, 34),
        "FC Dallas":              (44, 34),
        "Portland Timbers":       (44, 34),
        "Real Salt Lake":         (41, 34),
        "San Jose Earthquakes":   (41, 34),
        "Colorado Rapids":        (41, 34),
        "St. Louis City SC":      (32, 34),
        "LA Galaxy":              (30, 34),
        "Sporting Kansas City":   (28, 34),
        "Houston Dynamo FC":      (37, 34),
    },
}


# ─────────────────────────────────────────────────────────────
#  2025 Goals Data (for Poisson goal model)
#  Format: team -> (goals_for, goals_against, games_played)
# ─────────────────────────────────────────────────────────────

GOALS_2025: Dict[str, Tuple[int, int, int]] = {
    # Eastern Conference
    "Philadelphia Union":     (57, 35, 34),
    "FC Cincinnati":          (52, 40, 34),
    "Inter Miami CF":         (81, 55, 34),
    "Charlotte FC":           (55, 46, 34),
    "New York City FC":       (50, 44, 34),
    "Nashville SC":           (58, 45, 34),
    "Columbus Crew":          (55, 51, 34),
    "Chicago Fire FC":        (68, 60, 34),
    "Orlando City SC":        (63, 51, 34),
    "New York Red Bulls":     (48, 47, 34),
    "New England Revolution": (44, 51, 34),
    "Toronto FC":             (37, 44, 34),
    "CF Montréal":            (34, 60, 34),
    "Atlanta United FC":      (38, 63, 34),
    "D.C. United":            (30, 66, 34),
    # Western Conference
    "San Diego FC":           (64, 41, 34),
    "Vancouver Whitecaps FC": (66, 38, 34),
    "LAFC":                   (65, 40, 34),
    "Minnesota United FC":    (56, 39, 34),
    "Seattle Sounders FC":    (58, 48, 34),
    "Austin FC":              (37, 45, 34),
    "FC Dallas":              (52, 55, 34),
    "Portland Timbers":       (41, 48, 34),
    "Real Salt Lake":         (38, 49, 34),
    "San Jose Earthquakes":   (60, 63, 34),
    "Colorado Rapids":        (44, 56, 34),
    "Houston Dynamo FC":      (43, 56, 34),
    "St. Louis City SC":      (44, 58, 34),
    "LA Galaxy":              (46, 66, 34),
    "Sporting Kansas City":   (46, 70, 34),
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

def season_elo_from_points(pts: int, gp: int) -> float:
    pts_per_game = pts / gp
    avg_ppg = MLS_AVG_POINTS / 34.0
    delta = (pts_per_game - avg_ppg) * POINTS_TO_ELO_SCALE * 34.0
    return INITIAL_ELO + delta


def build_elo_ratings() -> Dict[str, float]:
    all_teams: set = set()
    for season_data in HISTORICAL_SEASONS.values():
        all_teams.update(season_data.keys())

    weighted_elo: Dict[str, List[Tuple[float, float]]] = {t: [] for t in all_teams}
    for season, season_data in HISTORICAL_SEASONS.items():
        weight = SEASON_WEIGHTS[season]
        for team, (pts, gp) in season_data.items():
            elo = season_elo_from_points(pts, gp)
            weighted_elo[team].append((elo, weight))

    ratings: Dict[str, float] = {}
    for team, pairs in weighted_elo.items():
        total_weight = sum(w for _, w in pairs)
        raw = sum(e * w for e, w in pairs) / total_weight
        regressed = INITIAL_ELO + (raw - INITIAL_ELO) * (1.0 - REGRESSION_TO_MEAN)
        ratings[team] = round(regressed, 1)
    return ratings


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))


def predict_three_way(
    home_elo: float,
    away_elo: float,
) -> Tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) using Elo + home advantage."""
    raw_home_win = expected_score(home_elo + HOME_ADVANTAGE, away_elo)
    evenness = 1.0 - abs(raw_home_win - 0.5) * 2.0
    p_draw = DRAW_BASE_RATE + DRAW_SENSITIVITY * evenness
    remaining = 1.0 - p_draw
    return raw_home_win * remaining, p_draw, (1.0 - raw_home_win) * remaining


# ─────────────────────────────────────────────────────────────
#  Poisson Goal Engine
# ─────────────────────────────────────────────────────────────

def build_goal_ratings(
    goals_data: Dict[str, Tuple[int, int, int]],
    league_avg: float = LEAGUE_AVG_GPG,
    regression: float = GOAL_REGRESSION,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute attack and defense strength ratings from goals data.

    attack[team] > 1.0  → above-average scorer
    defense[team] > 1.0 → above-average goals conceded (leaky defense)

    A 25 % regression toward 1.0 is applied to account for off-season
    squad changes and the uncertainty at the start of a new season.
    """
    attack: Dict[str, float] = {}
    defense: Dict[str, float] = {}
    for team, (gf, ga, gp) in goals_data.items():
        raw_att = (gf / gp) / league_avg
        raw_def = (ga / gp) / league_avg
        attack[team]  = 1.0 + (raw_att - 1.0) * (1.0 - regression)
        defense[team] = 1.0 + (raw_def - 1.0) * (1.0 - regression)
    return attack, defense


def expected_goals(
    home: str,
    away: str,
    attack: Dict[str, float],
    defense: Dict[str, float],
) -> Tuple[float, float]:
    """
    Expected goals for each team using the Dixon-Coles multiplicative model.

        λ_home = league_home_avg × home_attack × away_defense_weakness
        λ_away = league_away_avg × away_attack × home_defense_weakness

    Defaults to league-average strength (1.0) for unknown teams.
    """
    h_att  = attack.get(home,  1.0)
    a_def  = defense.get(away, 1.0)
    a_att  = attack.get(away,  1.0)
    h_def  = defense.get(home, 1.0)

    lam_home = LEAGUE_HOME_AVG * h_att * a_def
    lam_away = LEAGUE_AWAY_AVG * a_att * h_def
    return round(lam_home, 3), round(lam_away, 3)


def p_over_2_5(lam_home: float, lam_away: float) -> float:
    """
    Probability that total goals exceed 2.5 (i.e. 3 or more goals).

    Under the independence assumption, total goals T = G_home + G_away
    follows Poisson(λ_total) where λ_total = λ_home + λ_away.

        P(T ≤ 2) = e^{-λ} × (1 + λ + λ²/2)
        P(T > 2) = 1 − P(T ≤ 2)
    """
    lam = lam_home + lam_away
    p_le2 = math.exp(-lam) * (1.0 + lam + lam ** 2 / 2.0)
    return max(0.0, min(1.0, 1.0 - p_le2))


def p_exact_score_table(
    lam_home: float,
    lam_away: float,
    max_goals: int = 6,
) -> Dict[Tuple[int, int], float]:
    """
    Return a {(home_goals, away_goals): probability} table.
    Useful for inspecting the most likely scorelines.
    """
    table: Dict[Tuple[int, int], float] = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = (
                math.exp(-lam_home) * lam_home ** h / math.factorial(h)
                * math.exp(-lam_away) * lam_away ** a / math.factorial(a)
            )
            table[(h, a)] = p
    return table


def top_scorelines(
    lam_home: float,
    lam_away: float,
    n: int = 5,
) -> List[Tuple[Tuple[int, int], float]]:
    """Return the n most probable exact scorelines."""
    table = p_exact_score_table(lam_home, lam_away)
    return sorted(table.items(), key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────────────────────
#  Combined Prediction
# ─────────────────────────────────────────────────────────────

@dataclass
class MatchPrediction:
    fixture: Fixture
    # Elo model outputs
    home_elo: float
    away_elo: float
    p_home: float
    p_draw: float
    p_away: float
    # Goal model outputs
    xg_home: float
    xg_away: float
    p_over25: float

    @property
    def xg_total(self) -> float:
        return round(self.xg_home + self.xg_away, 2)

    @property
    def predicted_outcome(self) -> str:
        probs = {"Home Win": self.p_home, "Draw": self.p_draw, "Away Win": self.p_away}
        return max(probs, key=probs.__getitem__)

    @property
    def predicted_winner(self) -> str:
        o = self.predicted_outcome
        if o == "Home Win":
            return self.fixture.home
        if o == "Away Win":
            return self.fixture.away
        return "Draw"

    @property
    def over_under_call(self) -> str:
        return "OVER 2.5" if self.p_over25 >= 0.50 else "UNDER 2.5"

    @property
    def ou_confidence(self) -> str:
        p = self.p_over25 if self.p_over25 >= 0.50 else 1.0 - self.p_over25
        if p >= 0.70:
            return "HIGH"
        if p >= 0.58:
            return "MODERATE"
        return "SLIGHT"


def run_predictions() -> List[MatchPrediction]:
    elo_ratings = build_elo_ratings()
    attack, defense = build_goal_ratings(GOALS_2025)
    predictions: List[MatchPrediction] = []

    for fixture in ROUND_1_FIXTURES:
        h_elo = elo_ratings.get(fixture.home, INITIAL_ELO)
        a_elo = elo_ratings.get(fixture.away, INITIAL_ELO)
        p_home, p_draw, p_away = predict_three_way(h_elo, a_elo)
        xg_h, xg_a = expected_goals(fixture.home, fixture.away, attack, defense)
        p_o25 = p_over_2_5(xg_h, xg_a)
        predictions.append(MatchPrediction(
            fixture=fixture,
            home_elo=h_elo, away_elo=a_elo,
            p_home=p_home, p_draw=p_draw, p_away=p_away,
            xg_home=xg_h, xg_away=xg_a,
            p_over25=p_o25,
        ))
    return predictions


# ─────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────

W = 80
DIV = "─" * W
HDR = "═" * W


def print_report(predictions: List[MatchPrediction]) -> None:
    print(HDR)
    print("  MLS 2026 ROUND 1 — PREDICTIONS  |  Opening Weekend Feb 21-22")
    print("  Models: Elo (result) + Poisson/Dixon-Coles (goals) | Data: 2022-2025")
    print(HDR)

    current_date = ""
    for pred in predictions:
        if pred.fixture.date != current_date:
            current_date = pred.fixture.date
            label = "SATURDAY FEBRUARY 21" if "21" in current_date else "SUNDAY FEBRUARY 22"
            print(f"\n  ▶  {label}, 2026")
            print(DIV)

        h, a = pred.fixture.home, pred.fixture.away
        lam_h, lam_a = pred.xg_home, pred.xg_away
        scores = top_scorelines(lam_h, lam_a, n=3)

        print(f"\n  {pred.fixture.kickoff_et} ET  ·  {pred.fixture.broadcast}")
        print(f"  {h}  vs  {a}")
        print(f"  Elo: {pred.home_elo:.0f} vs {pred.away_elo:.0f}")
        print(
            f"  Result:  Home {pred.p_home:.1%}  ·  Draw {pred.p_draw:.1%}"
            f"  ·  Away {pred.p_away:.1%}  → {pred.predicted_winner}"
        )
        print(
            f"  Goals:   xG {lam_h:.2f} – {lam_a:.2f}  (total {pred.xg_total:.2f})"
            f"  →  Over 2.5: {pred.p_over25:.1%}  [{pred.ou_confidence} {pred.over_under_call}]"
        )
        score_str = "  ".join(f"{s[0]}-{s[1]} ({p:.1%})" for (s, p) in scores)
        print(f"  Top scorelines: {score_str}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{HDR}")
    print("  OVER / UNDER 2.5 GOALS SUMMARY — ALL 15 MATCHES")
    print(HDR)
    print(f"  {'Match':<46} {'xG':>6}  {'P(O2.5)':>8}  {'Call':<12}  Confidence")
    print(f"  {'─'*46}  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*8}")
    for pred in predictions:
        matchup = f"{pred.fixture.home} vs {pred.fixture.away}"
        if len(matchup) > 44:
            matchup = matchup[:44]
        call_str = "OVER ✓" if pred.over_under_call == "OVER 2.5" else "UNDER ✗"
        print(
            f"  {matchup:<46}  {pred.xg_total:>5.2f}  "
            f"{pred.p_over25:>7.1%}   {call_str:<12} {pred.ou_confidence}"
        )

    over_count = sum(1 for p in predictions if p.over_under_call == "OVER 2.5")
    under_count = len(predictions) - over_count
    print(f"\n  OVER count: {over_count}  |  UNDER count: {under_count}  (of {len(predictions)} matches)")

    # ── O/U ranked by probability ──────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  RANKED BY OVER 2.5 PROBABILITY (highest → lowest)")
    print(DIV)
    ranked = sorted(predictions, key=lambda p: p.p_over25, reverse=True)
    for i, pred in enumerate(ranked, 1):
        matchup = f"{pred.fixture.home} vs {pred.fixture.away}"
        bar_len = int(pred.p_over25 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i:>2}. {matchup:<42} {pred.p_over25:>5.1%}  {bar}")

    # ── Elo leaderboard ────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  TEAM ELO RATINGS  |  ATTACK ↑ / DEFENSE ↓ STRENGTHS (2025)")
    print(DIV)
    attack, defense = build_goal_ratings(GOALS_2025)
    elo_ratings = build_elo_ratings()
    all_teams = {p.fixture.home for p in predictions} | {p.fixture.away for p in predictions}
    table = sorted(
        [(t, elo_ratings.get(t, 1500.0), attack.get(t, 1.0), defense.get(t, 1.0))
         for t in all_teams],
        key=lambda x: x[1], reverse=True
    )
    print(f"  {'Team':<32} {'Elo':>5}  {'Att':>5}  {'Def':>5}  (Att>1=good scorer, Def<1=good defender)")
    print(f"  {'─'*32}  {'─'*5}  {'─'*5}  {'─'*5}")
    for team, elo, att, def_ in table:
        att_bar = "▲" if att > 1.05 else ("▼" if att < 0.95 else "─")
        def_bar = "▼" if def_ < 0.95 else ("▲" if def_ > 1.05 else "─")
        print(f"  {team:<32} {elo:>5.0f}  {att:>5.2f}{att_bar}  {def_:>5.2f}{def_bar}")

    print(f"\n{HDR}")
    print("  METHODOLOGY NOTES")
    print(f"  {'─'*76}")
    print(
        "  • Result probabilities: Elo model, seasons 2022-2025 (50/25/15/10 % weights),\n"
        "    +100 Elo home advantage, 30 % regression to mean between seasons.\n"
        "  • Over/Under: Independent Poisson (Dixon-Coles style). Attack/defense\n"
        "    strengths from 2025 GF/GA, regressed 25 % toward league average.\n"
        "    League home avg 1.60 g/g, away avg 1.30 g/g (2025 MLS: 2.90 g/g).\n"
        "  • Does NOT factor in: injuries, new signings, weather, referee, or\n"
        "    preseason form. Use alongside qualitative analysis."
    )
    print(HDR)


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preds = run_predictions()
    print_report(preds)

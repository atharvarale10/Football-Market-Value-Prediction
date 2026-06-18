from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "football_player_market_values.csv"
MODEL_PATH = MODEL_DIR / "market_value_model.joblib"

NUMERIC_FEATURES = [
    "age",
    "height_cm",
    "overall_rating",
    "potential_rating",
    "minutes_last_season",
    "goals_last_season",
    "assists_last_season",
    "xg_per90",
    "xa_per90",
    "progressive_actions_per90",
    "pressures_per90",
    "fitness_score",
    "injury_days_last_12m",
    "matches_missed_last_12m",
    "contract_years_left",
    "wage_eur_week",
    "international_caps",
    "champions_league_minutes",
    "club_reputation",
    "league_strength",
    "form_index",
    "sell_on_potential",
    "release_clause_ratio",
]

CATEGORICAL_FEATURES = ["position", "dominant_foot", "nationality_region"]
TARGET = "market_value_eur"

FIRST_NAMES = [
    "Mateo",
    "Luca",
    "Noah",
    "Elias",
    "Theo",
    "Rafael",
    "Daniel",
    "Nico",
    "Ivan",
    "Hugo",
    "Milan",
    "Adam",
    "Leo",
    "Oscar",
    "Felix",
    "Arda",
    "Yusuf",
    "Karim",
    "Enzo",
    "Alex",
]

LAST_NAMES = [
    "Silva",
    "Martinez",
    "Kovac",
    "Moreau",
    "Rossi",
    "Fernandez",
    "Diallo",
    "Aydin",
    "Bennett",
    "Santos",
    "Ndiaye",
    "Schmidt",
    "Costa",
    "Petrov",
    "Garcia",
    "Okafor",
    "Dubois",
    "Hansen",
    "Mendes",
    "Popescu",
]

POSITIONS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
POSITION_WEIGHTS = np.array([0.10, 0.17, 0.14, 0.11, 0.16, 0.11, 0.12, 0.09])
POSITION_PREMIUM = {
    "GK": -0.10,
    "CB": -0.02,
    "FB": 0.02,
    "DM": 0.00,
    "CM": 0.06,
    "AM": 0.14,
    "W": 0.18,
    "ST": 0.22,
}

REGIONS = ["EU", "South America", "Africa", "UK/Ireland", "North America", "Asia"]
REGION_WEIGHTS = np.array([0.46, 0.20, 0.17, 0.08, 0.05, 0.04])

REAL_PLAYER_MARKET_VALUES = [
    {"rank": 1, "player_name": "Lamine Yamal", "club": "FC Barcelona", "position_detail": "Right Winger", "age": 18, "nationality_region": "EU", "market_value_eur": 200_000_000},
    {"rank": 2, "player_name": "Erling Haaland", "club": "Manchester City", "position_detail": "Centre-Forward", "age": 25, "nationality_region": "EU", "market_value_eur": 200_000_000},
    {"rank": 3, "player_name": "Kylian Mbappe", "club": "Real Madrid", "position_detail": "Centre-Forward", "age": 27, "nationality_region": "EU", "market_value_eur": 180_000_000},
    {"rank": 4, "player_name": "Pedri", "club": "FC Barcelona", "position_detail": "Central Midfield", "age": 23, "nationality_region": "EU", "market_value_eur": 150_000_000},
    {"rank": 5, "player_name": "Michael Olise", "club": "Bayern Munich", "position_detail": "Right Winger", "age": 24, "nationality_region": "EU", "market_value_eur": 150_000_000},
    {"rank": 6, "player_name": "Joao Neves", "club": "Paris Saint-Germain", "position_detail": "Central Midfield", "age": 21, "nationality_region": "EU", "market_value_eur": 140_000_000},
    {"rank": 7, "player_name": "Khvicha Kvaratskhelia", "club": "Paris Saint-Germain", "position_detail": "Left Winger", "age": 25, "nationality_region": "EU", "market_value_eur": 140_000_000},
    {"rank": 8, "player_name": "Vitinha", "club": "Paris Saint-Germain", "position_detail": "Defensive Midfield", "age": 26, "nationality_region": "EU", "market_value_eur": 140_000_000},
    {"rank": 9, "player_name": "Vinicius Junior", "club": "Real Madrid", "position_detail": "Left Winger", "age": 25, "nationality_region": "South America", "market_value_eur": 140_000_000},
    {"rank": 10, "player_name": "Jude Bellingham", "club": "Real Madrid", "position_detail": "Attacking Midfield", "age": 22, "nationality_region": "UK/Ireland", "market_value_eur": 130_000_000},
    {"rank": 11, "player_name": "Desire Doue", "club": "Paris Saint-Germain", "position_detail": "Right Winger", "age": 21, "nationality_region": "EU", "market_value_eur": 120_000_000},
    {"rank": 12, "player_name": "Declan Rice", "club": "Arsenal FC", "position_detail": "Central Midfield", "age": 27, "nationality_region": "UK/Ireland", "market_value_eur": 120_000_000},
    {"rank": 13, "player_name": "Bukayo Saka", "club": "Arsenal FC", "position_detail": "Right Winger", "age": 24, "nationality_region": "UK/Ireland", "market_value_eur": 110_000_000},
    {"rank": 14, "player_name": "Moises Caicedo", "club": "Chelsea FC", "position_detail": "Defensive Midfield", "age": 24, "nationality_region": "South America", "market_value_eur": 100_000_000},
    {"rank": 15, "player_name": "Fermin Lopez", "club": "FC Barcelona", "position_detail": "Attacking Midfield", "age": 23, "nationality_region": "EU", "market_value_eur": 100_000_000},
    {"rank": 16, "player_name": "Florian Wirtz", "club": "Liverpool FC", "position_detail": "Attacking Midfield", "age": 23, "nationality_region": "EU", "market_value_eur": 100_000_000},
    {"rank": 17, "player_name": "Jamal Musiala", "club": "Bayern Munich", "position_detail": "Attacking Midfield", "age": 23, "nationality_region": "EU", "market_value_eur": 100_000_000},
    {"rank": 18, "player_name": "Julian Alvarez", "club": "Atletico de Madrid", "position_detail": "Centre-Forward", "age": 26, "nationality_region": "South America", "market_value_eur": 100_000_000},
    {"rank": 19, "player_name": "Cole Palmer", "club": "Chelsea FC", "position_detail": "Attacking Midfield", "age": 24, "nationality_region": "UK/Ireland", "market_value_eur": 100_000_000},
    {"rank": 20, "player_name": "William Saliba", "club": "Arsenal FC", "position_detail": "Centre-Back", "age": 25, "nationality_region": "EU", "market_value_eur": 100_000_000},
    {"rank": 21, "player_name": "Dominik Szoboszlai", "club": "Liverpool FC", "position_detail": "Attacking Midfield", "age": 25, "nationality_region": "EU", "market_value_eur": 100_000_000},
    {"rank": 22, "player_name": "Ousmane Dembele", "club": "Paris Saint-Germain", "position_detail": "Centre-Forward", "age": 29, "nationality_region": "EU", "market_value_eur": 100_000_000},
    {"rank": 23, "player_name": "Yan Diomande", "club": "RB Leipzig", "position_detail": "Left Winger", "age": 19, "nationality_region": "Africa", "market_value_eur": 90_000_000},
    {"rank": 24, "player_name": "Arda Guler", "club": "Real Madrid", "position_detail": "Attacking Midfield", "age": 21, "nationality_region": "EU", "market_value_eur": 90_000_000},
    {"rank": 25, "player_name": "Aleksandar Pavlovic", "club": "Bayern Munich", "position_detail": "Defensive Midfield", "age": 22, "nationality_region": "EU", "market_value_eur": 90_000_000},
    {"rank": 26, "player_name": "Enzo Fernandez", "club": "Chelsea FC", "position_detail": "Central Midfield", "age": 25, "nationality_region": "South America", "market_value_eur": 90_000_000},
    {"rank": 27, "player_name": "Rayan Cherki", "club": "Manchester City", "position_detail": "Attacking Midfield", "age": 22, "nationality_region": "EU", "market_value_eur": 90_000_000},
    {"rank": 28, "player_name": "Morgan Rogers", "club": "Aston Villa", "position_detail": "Attacking Midfield", "age": 23, "nationality_region": "UK/Ireland", "market_value_eur": 90_000_000},
    {"rank": 29, "player_name": "Federico Valverde", "club": "Real Madrid", "position_detail": "Central Midfield", "age": 27, "nationality_region": "South America", "market_value_eur": 90_000_000},
    {"rank": 30, "player_name": "Lautaro Martinez", "club": "Inter Milan", "position_detail": "Centre-Forward", "age": 28, "nationality_region": "South America", "market_value_eur": 85_000_000},
    {"rank": 31, "player_name": "Alexander Isak", "club": "Liverpool FC", "position_detail": "Centre-Forward", "age": 26, "nationality_region": "EU", "market_value_eur": 85_000_000},
    {"rank": 32, "player_name": "Estevao", "club": "Chelsea FC", "position_detail": "Right Winger", "age": 19, "nationality_region": "South America", "market_value_eur": 80_000_000},
    {"rank": 33, "player_name": "Pau Cubarsi", "club": "FC Barcelona", "position_detail": "Centre-Back", "age": 19, "nationality_region": "EU", "market_value_eur": 80_000_000},
    {"rank": 34, "player_name": "Nico Paz", "club": "Como 1907", "position_detail": "Attacking Midfield", "age": 21, "nationality_region": "South America", "market_value_eur": 80_000_000},
    {"rank": 35, "player_name": "Warren Zaire-Emery", "club": "Paris Saint-Germain", "position_detail": "Central Midfield", "age": 20, "nationality_region": "EU", "market_value_eur": 80_000_000},
    {"rank": 36, "player_name": "Hugo Ekitike", "club": "Liverpool FC", "position_detail": "Centre-Forward", "age": 23, "nationality_region": "EU", "market_value_eur": 80_000_000},
    {"rank": 37, "player_name": "Willian Pacho", "club": "Paris Saint-Germain", "position_detail": "Centre-Back", "age": 24, "nationality_region": "South America", "market_value_eur": 80_000_000},
    {"rank": 38, "player_name": "Joao Pedro", "club": "Chelsea FC", "position_detail": "Centre-Forward", "age": 24, "nationality_region": "South America", "market_value_eur": 80_000_000},
    {"rank": 39, "player_name": "Nuno Mendes", "club": "Paris Saint-Germain", "position_detail": "Left-Back", "age": 23, "nationality_region": "EU", "market_value_eur": 80_000_000},
    {"rank": 40, "player_name": "Antoine Semenyo", "club": "Manchester City", "position_detail": "Right Winger", "age": 26, "nationality_region": "Africa", "market_value_eur": 80_000_000},
    {"rank": 41, "player_name": "Ryan Gravenberch", "club": "Liverpool FC", "position_detail": "Defensive Midfield", "age": 24, "nationality_region": "EU", "market_value_eur": 80_000_000},
    {"rank": 42, "player_name": "Achraf Hakimi", "club": "Paris Saint-Germain", "position_detail": "Right-Back", "age": 27, "nationality_region": "Africa", "market_value_eur": 80_000_000},
    {"rank": 43, "player_name": "Sandro Tonali", "club": "Newcastle United", "position_detail": "Defensive Midfield", "age": 26, "nationality_region": "EU", "market_value_eur": 80_000_000},
    {"rank": 44, "player_name": "Kenan Yildiz", "club": "Juventus FC", "position_detail": "Left Winger", "age": 21, "nationality_region": "EU", "market_value_eur": 75_000_000},
    {"rank": 45, "player_name": "Benjamin Sesko", "club": "Manchester United", "position_detail": "Centre-Forward", "age": 23, "nationality_region": "EU", "market_value_eur": 75_000_000},
    {"rank": 46, "player_name": "Elliot Anderson", "club": "Nottingham Forest", "position_detail": "Central Midfield", "age": 23, "nationality_region": "UK/Ireland", "market_value_eur": 75_000_000},
    {"rank": 47, "player_name": "Matheus Cunha", "club": "Manchester United", "position_detail": "Centre-Forward", "age": 27, "nationality_region": "South America", "market_value_eur": 75_000_000},
    {"rank": 48, "player_name": "Jeremy Doku", "club": "Manchester City", "position_detail": "Left Winger", "age": 24, "nationality_region": "EU", "market_value_eur": 75_000_000},
    {"rank": 49, "player_name": "Gabriel", "club": "Arsenal FC", "position_detail": "Centre-Back", "age": 28, "nationality_region": "South America", "market_value_eur": 75_000_000},
    {"rank": 50, "player_name": "Martin Zubimendi", "club": "Arsenal FC", "position_detail": "Defensive Midfield", "age": 27, "nationality_region": "EU", "market_value_eur": 75_000_000},
]

POSITION_DETAIL_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB",
    "Right-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Attacking Midfield": "AM",
    "Left Winger": "W",
    "Right Winger": "W",
    "Centre-Forward": "ST",
}


def bounded(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def position_rate(position: str, rates: dict[str, float]) -> float:
    return rates.get(position, rates["CM"])


def make_dataset(rows: int = 7500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = rng.choice(POSITIONS, rows, p=POSITION_WEIGHTS / POSITION_WEIGHTS.sum())
    regions = rng.choice(REGIONS, rows, p=REGION_WEIGHTS / REGION_WEIGHTS.sum())

    age = np.rint(16 + rng.beta(2.15, 3.2, rows) * 22).astype(int)
    height_base = np.array(
        [
            {"GK": 190, "CB": 187, "FB": 178, "DM": 181, "CM": 179, "AM": 176, "W": 175, "ST": 183}[p]
            for p in positions
        ]
    )
    height_cm = bounded(rng.normal(height_base, 5.5), 162, 202).round(0)

    league_strength = rng.choice([54, 62, 69, 76, 84, 91, 96], rows, p=[0.08, 0.13, 0.20, 0.22, 0.20, 0.12, 0.05])
    club_reputation = bounded(league_strength + rng.normal(0, 9, rows), 35, 100)

    raw_talent = bounded(rng.normal(66, 10, rows) + (club_reputation - 70) * 0.14, 42, 96)
    age_curve = -0.085 * np.maximum(age - 27, 0) ** 1.35 + 0.32 * np.maximum(22 - np.abs(age - 22), 0)
    potential_rating = bounded(raw_talent + rng.normal(6.5, 4.5, rows) - np.maximum(age - 25, 0) * 0.95, 48, 98)
    overall_rating = bounded(raw_talent + age_curve + rng.normal(0, 3.5, rows), 43, 94)
    overall_rating = np.minimum(overall_rating, potential_rating + bounded(rng.normal(1.0, 2.0, rows), -2, 5))

    fitness_score = bounded(rng.normal(82, 9, rows) - np.maximum(age - 30, 0) * 1.4, 38, 99)
    injury_days_last_12m = bounded(rng.gamma(1.35, 15, rows) + np.maximum(age - 30, 0) * rng.uniform(0, 4, rows), 0, 210)
    matches_missed_last_12m = np.rint(injury_days_last_12m / rng.uniform(5.7, 8.4, rows)).astype(int)

    starter_score = bounded((overall_rating - 58) / 25 + (fitness_score - 70) / 80 + rng.normal(0, 0.18, rows), 0.02, 1.0)
    minutes_last_season = np.rint(bounded(3500 * starter_score - injury_days_last_12m * 7 + rng.normal(0, 270, rows), 80, 4050)).astype(int)

    goals_rate = np.array(
        [
            position_rate(
                p,
                {"GK": 0.00, "CB": 0.04, "FB": 0.03, "DM": 0.04, "CM": 0.07, "AM": 0.16, "W": 0.22, "ST": 0.36},
            )
            for p in positions
        ]
    )
    assists_rate = np.array(
        [
            position_rate(
                p,
                {"GK": 0.00, "CB": 0.02, "FB": 0.10, "DM": 0.07, "CM": 0.12, "AM": 0.22, "W": 0.24, "ST": 0.10},
            )
            for p in positions
        ]
    )
    attacking_quality = bounded((overall_rating - 58) / 28 + rng.normal(0, 0.15, rows), 0.05, 1.6)
    full_90s = minutes_last_season / 90
    goals_last_season = rng.poisson(np.maximum(full_90s * goals_rate * attacking_quality, 0.01))
    assists_last_season = rng.poisson(np.maximum(full_90s * assists_rate * attacking_quality, 0.01))

    xg_per90 = bounded(goals_rate * attacking_quality + rng.normal(0.03, 0.05, rows), 0, 0.9)
    xa_per90 = bounded(assists_rate * attacking_quality + rng.normal(0.03, 0.04, rows), 0, 0.65)
    progressive_actions_per90 = bounded(
        np.array(
            [
                position_rate(
                    p,
                    {"GK": 0.5, "CB": 2.0, "FB": 4.8, "DM": 3.4, "CM": 5.2, "AM": 6.4, "W": 7.1, "ST": 2.8},
                )
                for p in positions
            ]
        )
        * bounded((overall_rating - 50) / 30, 0.35, 1.55)
        + rng.normal(0, 0.8, rows),
        0,
        11.5,
    )
    pressures_per90 = bounded(
        np.array(
            [
                position_rate(
                    p,
                    {"GK": 0.3, "CB": 8.0, "FB": 15.0, "DM": 18.0, "CM": 19.0, "AM": 17.0, "W": 14.0, "ST": 12.0},
                )
                for p in positions
            ]
        )
        * bounded(fitness_score / 84, 0.55, 1.25)
        + rng.normal(0, 2.2, rows),
        0,
        32,
    )

    contract_years_left = bounded(rng.gamma(2.2, 0.95, rows), 0.1, 5.7).round(1)
    wage_eur_week = np.exp(7.7 + (overall_rating - 55) * 0.083 + (club_reputation - 65) * 0.017 + rng.normal(0, 0.42, rows))
    wage_eur_week = bounded(wage_eur_week, 800, 680000).round(-2)
    international_caps = rng.poisson(
        np.maximum((overall_rating - 68) * 0.42 + (age - 20) * 0.20 + (league_strength - 70) * 0.05, 0.03)
    )
    champions_league_minutes = np.rint(
        bounded(
            (club_reputation - 74) * 23 + (overall_rating - 70) * 18 + rng.normal(0, 190, rows),
            0,
            1300,
        )
    ).astype(int)
    form_index = bounded(
        55
        + (goals_last_season * 1.35 + assists_last_season * 1.1)
        + (overall_rating - 65) * 0.9
        - injury_days_last_12m * 0.09
        + rng.normal(0, 8, rows),
        10,
        99,
    )
    sell_on_potential = bounded(
        100 - age * 2.2 + (potential_rating - overall_rating) * 4.6 + rng.normal(0, 8, rows),
        0,
        100,
    )
    release_clause_ratio = bounded(rng.normal(1.7, 0.38, rows) + (contract_years_left - 2.0) * 0.06, 1.0, 3.2)

    pos_premium = np.array([POSITION_PREMIUM[p] for p in positions])
    elite_output = goals_last_season * 0.035 + assists_last_season * 0.027
    age_value_curve = -0.021 * (age - 24.5) ** 2
    scarcity = np.where(np.isin(positions, ["ST", "W", "AM"]), 0.16, np.where(positions == "GK", -0.08, 0.02))
    log_value = (
        0.85
        + overall_rating * 0.085
        + potential_rating * 0.052
        + age_value_curve
        + np.log1p(minutes_last_season) * 0.17
        + elite_output
        + xg_per90 * 0.62
        + xa_per90 * 0.55
        + progressive_actions_per90 * 0.035
        + fitness_score * 0.009
        - injury_days_last_12m * 0.0045
        + contract_years_left * 0.105
        + np.log1p(wage_eur_week) * 0.18
        + international_caps * 0.013
        + champions_league_minutes * 0.00028
        + club_reputation * 0.017
        + league_strength * 0.012
        + form_index * 0.011
        + sell_on_potential * 0.009
        + (release_clause_ratio - 1.0) * 0.11
        + pos_premium
        + scarcity
        + rng.normal(0, 0.34, rows)
    )

    raw_market_value = np.exp(log_value) / 92
    market_value_eur = bounded(
        raw_market_value / (1 + raw_market_value / 280_000_000),
        75_000,
        72_000_000,
    ).round(-4)

    names = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        for _ in range(rows)
    ]
    clubs = [
        f"{tier} {suffix}"
        for tier, suffix in zip(
            rng.choice(["Athletic", "United", "Sporting", "Racing", "City", "Real", "Inter", "Olympique"], rows),
            rng.choice(["FC", "CF", "SC", "Town", "Club"], rows),
        )
    ]

    df = pd.DataFrame(
        {
            "player_name": names,
            "club": clubs,
            "position": positions,
            "dominant_foot": rng.choice(["Right", "Left", "Both"], rows, p=[0.68, 0.24, 0.08]),
            "nationality_region": regions,
            "age": age,
            "height_cm": height_cm.astype(int),
            "overall_rating": overall_rating.round(1),
            "potential_rating": potential_rating.round(1),
            "minutes_last_season": minutes_last_season,
            "goals_last_season": goals_last_season,
            "assists_last_season": assists_last_season,
            "xg_per90": xg_per90.round(2),
            "xa_per90": xa_per90.round(2),
            "progressive_actions_per90": progressive_actions_per90.round(2),
            "pressures_per90": pressures_per90.round(2),
            "fitness_score": fitness_score.round(1),
            "injury_days_last_12m": injury_days_last_12m.round(0).astype(int),
            "matches_missed_last_12m": matches_missed_last_12m,
            "contract_years_left": contract_years_left,
            "wage_eur_week": wage_eur_week.astype(int),
            "international_caps": international_caps,
            "champions_league_minutes": champions_league_minutes,
            "club_reputation": club_reputation.round(1),
            "league_strength": league_strength,
            "form_index": form_index.round(1),
            "sell_on_potential": sell_on_potential.round(1),
            "release_clause_ratio": release_clause_ratio.round(2),
            TARGET: market_value_eur.astype(int),
        }
    )
    df["data_source"] = "synthetic_training_sample"
    df["valuation_source"] = "model_generated"
    df["source_rank"] = pd.NA
    df["source_url"] = ""

    real_df = make_real_player_rows(seed=seed + 7)
    return pd.concat([real_df, df], ignore_index=True).sort_values(TARGET, ascending=False).reset_index(drop=True)


def make_real_player_rows(seed: int = 49) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item in REAL_PLAYER_MARKET_VALUES:
        position = POSITION_DETAIL_MAP[item["position_detail"]]
        value_m = item["market_value_eur"] / 1_000_000
        elite = np.log1p(value_m) / np.log1p(220)
        age = item["age"]
        young_upside = max(0, 24 - age)

        overall = bounded(np.array([74 + elite * 20 + rng.normal(0, 1.2)]), 68, 94)[0]
        potential = bounded(np.array([overall + young_upside * 1.25 + rng.normal(1.2, 1.0)]), overall, 99)[0]
        fitness = bounded(np.array([88 - max(0, age - 27) * 0.8 + rng.normal(0, 3)]), 70, 98)[0]
        injury_days = int(bounded(np.array([rng.gamma(1.2, 9) + max(0, age - 28) * 4]), 0, 95)[0])
        minutes = int(bounded(np.array([2350 + elite * 1150 - injury_days * 5 + rng.normal(0, 180)]), 900, 4050)[0])
        full_90s = minutes / 90

        goals_rate = position_rate(
            position,
            {"GK": 0.00, "CB": 0.04, "FB": 0.04, "DM": 0.05, "CM": 0.08, "AM": 0.17, "W": 0.24, "ST": 0.43},
        )
        assists_rate = position_rate(
            position,
            {"GK": 0.00, "CB": 0.02, "FB": 0.12, "DM": 0.08, "CM": 0.13, "AM": 0.22, "W": 0.24, "ST": 0.11},
        )
        output_multiplier = 0.9 + elite * 0.55
        goals = int(round(max(0, full_90s * goals_rate * output_multiplier + rng.normal(0, 1.6))))
        assists = int(round(max(0, full_90s * assists_rate * output_multiplier + rng.normal(0, 1.4))))

        height = {
            "GK": 190,
            "CB": 188,
            "FB": 179,
            "DM": 181,
            "CM": 178,
            "AM": 176,
            "W": 177,
            "ST": 184,
        }[position] + rng.normal(0, 3)

        rows.append(
            {
                "player_name": item["player_name"],
                "club": item["club"],
                "position": position,
                "dominant_foot": "Left" if item["player_name"] in {"Lamine Yamal", "Bukayo Saka", "Cole Palmer", "Michael Olise", "Arda Guler"} else "Right",
                "nationality_region": item["nationality_region"],
                "age": age,
                "height_cm": int(round(bounded(np.array([height]), 165, 202)[0])),
                "overall_rating": round(float(overall), 1),
                "potential_rating": round(float(potential), 1),
                "minutes_last_season": minutes,
                "goals_last_season": goals,
                "assists_last_season": assists,
                "xg_per90": round(float(bounded(np.array([goals_rate * output_multiplier + rng.normal(0, 0.03)]), 0, 0.9)[0]), 2),
                "xa_per90": round(float(bounded(np.array([assists_rate * output_multiplier + rng.normal(0, 0.03)]), 0, 0.65)[0]), 2),
                "progressive_actions_per90": round(float(bounded(np.array([2.0 + elite * 5.8 + (1.4 if position in {"W", "AM", "CM"} else 0) + rng.normal(0, 0.5)]), 0, 11.5)[0]), 2),
                "pressures_per90": round(float(bounded(np.array([10 + elite * 10 + (3 if position in {"DM", "CM", "AM"} else 0) + rng.normal(0, 1.6)]), 0, 32)[0]), 2),
                "fitness_score": round(float(fitness), 1),
                "injury_days_last_12m": injury_days,
                "matches_missed_last_12m": int(round(injury_days / 7)),
                "contract_years_left": round(float(bounded(np.array([2.1 + young_upside * 0.16 + rng.normal(0, 0.75)]), 0.4, 5.7)[0]), 1),
                "wage_eur_week": int(round(bounded(np.array([35_000 + value_m * 2050 + rng.normal(0, 25_000)]), 20_000, 680_000)[0], -2)),
                "international_caps": int(round(bounded(np.array([elite * 70 + max(0, age - 22) * 2 + rng.normal(0, 8)]), 0, 140)[0])),
                "champions_league_minutes": int(round(bounded(np.array([elite * 1000 + rng.normal(0, 160)]), 0, 1300)[0])),
                "club_reputation": round(float(bounded(np.array([88 + elite * 11 + rng.normal(0, 2)]), 72, 100)[0]), 1),
                "league_strength": int(round(bounded(np.array([86 + elite * 10 + rng.normal(0, 2)]), 75, 100)[0])),
                "form_index": round(float(bounded(np.array([68 + elite * 26 + rng.normal(0, 3)]), 55, 99)[0]), 1),
                "sell_on_potential": round(float(bounded(np.array([35 + young_upside * 9 + elite * 35 + rng.normal(0, 4)]), 15, 100)[0]), 1),
                "release_clause_ratio": round(float(bounded(np.array([1.55 + elite * 0.65 + rng.normal(0, 0.16)]), 1.0, 3.2)[0]), 2),
                TARGET: item["market_value_eur"],
                "data_source": "real_transfermarkt_top50",
                "valuation_source": "Transfermarkt current market value snapshot",
                "source_rank": item["rank"],
                "source_url": "https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop",
            }
        )
    return pd.DataFrame(rows)


def train(rows: int = 7500, seed: int = 42) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = make_dataset(rows=rows, seed=seed)
    df.to_csv(DATA_PATH, index=False)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = np.log1p(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=seed)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = GradientBoostingRegressor(
        n_estimators=380,
        learning_rate=0.035,
        max_depth=3,
        min_samples_leaf=8,
        subsample=0.86,
        random_state=seed,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    pred_eur = np.expm1(pred_log)
    actual_eur = np.expm1(y_test)
    metrics = {
        "r2_log_value": round(float(r2_score(y_test, pred_log)), 3),
        "mae_eur": round(float(mean_absolute_error(actual_eur, pred_eur)), 0),
        "median_market_value_eur": int(df[TARGET].median()),
        "rows": len(df),
    }

    artifact = {
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "training_data_path": str(DATA_PATH),
    }
    joblib.dump(artifact, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    result = train()
    print(f"Saved dataset to {DATA_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    print(result)

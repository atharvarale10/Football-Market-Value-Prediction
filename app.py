from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import CATEGORICAL_FEATURES, DATA_PATH, MODEL_PATH, NUMERIC_FEATURES, TARGET, train


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Football Market Value Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


MARKET_CSS = """
<style>
    :root {
        --ink: #17202a;
        --muted: #667085;
        --line: #d8dee9;
        --surface: #ffffff;
        --panel: #f7f9fc;
        --navy: #10233f;
        --green: #0f8b6f;
        --red: #c2410c;
        --gold: #b7791f;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(16,35,63,0.08), rgba(255,255,255,0.88) 260px),
            #f2f5f8;
        color: var(--ink);
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 10px 24px rgba(16,35,63,0.05);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }
    .metric-card {
        min-height: 104px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(16,35,63,0.05);
    }
    .metric-label {
        color: #475467;
        font-size: 0.86rem;
        line-height: 1.2;
        margin-bottom: 11px;
    }
    .metric-number {
        color: #1d2939;
        font-size: 1.42rem;
        line-height: 1.12;
        font-weight: 650;
        overflow-wrap: anywhere;
    }
    @media (max-width: 1100px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 640px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    section[data-testid="stSidebar"] {
        background: #edf2f7;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stTabs"] button {
        font-weight: 650;
    }
    .market-band {
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.92);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 12px 30px rgba(16,35,63,0.06);
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    .tag {
        display: inline-block;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 3px 9px;
        margin-right: 6px;
        background: #fff;
        font-size: 0.8rem;
        color: #344054;
    }
</style>
"""
st.markdown(MARKET_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict:
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        train()
    return pd.read_csv(DATA_PATH)


def eur(value: float) -> str:
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}K"
    return f"EUR {value:,.0f}"


def clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def predict_value(artifact: dict, features: dict) -> float:
    row = pd.DataFrame([{key: features[key] for key in NUMERIC_FEATURES + CATEGORICAL_FEATURES}])
    return float(np.expm1(artifact["pipeline"].predict(row)[0]))


def uncertainty_band(value: float, features: dict) -> tuple[float, float]:
    risk = 0.13
    risk += 0.07 if features["age"] <= 20 else 0
    risk += 0.05 if features["age"] >= 31 else 0
    risk += min(features["injury_days_last_12m"] / 700, 0.10)
    risk += 0.05 if features["contract_years_left"] < 1 else 0
    risk += 0.04 if features["minutes_last_season"] < 900 else 0
    return value * (1 - risk), value * (1 + risk)


def future_frame(artifact: dict, base_features: dict, years: int, scenario: str) -> pd.DataFrame:
    rows = []
    for year in range(0, years + 1):
        f = dict(base_features)
        f["age"] = int(base_features["age"] + year)
        f["contract_years_left"] = round(clamp(base_features["contract_years_left"] - year, 0.1, 5.7), 1)

        if scenario == "Breakout":
            growth = max(0, 24 - f["age"]) * 0.55 + 1.2
            f["overall_rating"] = clamp(base_features["overall_rating"] + growth * year, 43, 96)
            f["potential_rating"] = clamp(base_features["potential_rating"] + 0.35 * year, 48, 99)
            f["minutes_last_season"] = int(clamp(base_features["minutes_last_season"] + 230 * year, 80, 4050))
            f["form_index"] = clamp(base_features["form_index"] + 4.5 * year, 10, 99)
            f["fitness_score"] = clamp(base_features["fitness_score"] + 1.0 * year, 38, 99)
            f["injury_days_last_12m"] = int(clamp(base_features["injury_days_last_12m"] - 7 * year, 0, 210))
        elif scenario == "Injury Risk":
            f["overall_rating"] = clamp(base_features["overall_rating"] - 0.7 * year, 43, 96)
            f["minutes_last_season"] = int(clamp(base_features["minutes_last_season"] - 250 * year, 80, 4050))
            f["fitness_score"] = clamp(base_features["fitness_score"] - 3.0 * year, 38, 99)
            f["injury_days_last_12m"] = int(clamp(base_features["injury_days_last_12m"] + 18 * year, 0, 210))
            f["matches_missed_last_12m"] = int(clamp(base_features["matches_missed_last_12m"] + 3 * year, 0, 38))
            f["form_index"] = clamp(base_features["form_index"] - 3.5 * year, 10, 99)
        elif scenario == "Prime Stability":
            age_drag = max(0, f["age"] - 29) * 0.8
            f["overall_rating"] = clamp(base_features["overall_rating"] + min(year, 2) * 0.55 - age_drag, 43, 96)
            f["potential_rating"] = clamp(max(f["overall_rating"], base_features["potential_rating"] - 0.45 * year), 48, 99)
            f["minutes_last_season"] = int(clamp(base_features["minutes_last_season"] + 80 * min(year, 2), 80, 4050))
            f["form_index"] = clamp(base_features["form_index"] + 1.0 * min(year, 2) - age_drag, 10, 99)
        else:
            decline = max(0, f["age"] - 30) * 1.2
            f["overall_rating"] = clamp(base_features["overall_rating"] - decline - 0.25 * year, 43, 96)
            f["potential_rating"] = clamp(max(f["overall_rating"], base_features["potential_rating"] - 0.9 * year), 48, 99)
            f["minutes_last_season"] = int(clamp(base_features["minutes_last_season"] - 140 * year, 80, 4050))
            f["form_index"] = clamp(base_features["form_index"] - 1.8 * year - decline, 10, 99)

        f["sell_on_potential"] = clamp(base_features["sell_on_potential"] - 5.8 * year + max(0, 23 - f["age"]) * 0.8, 0, 100)
        f["wage_eur_week"] = int(clamp(base_features["wage_eur_week"] * (1 + 0.055 * year), 800, 680000))
        f["market_value_eur"] = predict_value(artifact, f)
        f["year"] = f"+{year}Y"
        rows.append(f)
    return pd.DataFrame(rows)


def build_features_from_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Player Profile")
    preset_options = ["Custom"] + df.head(250)["player_name"].tolist()
    preset = st.sidebar.selectbox("Start from market player", preset_options)
    if preset != "Custom":
        player = df.loc[df["player_name"] == preset].iloc[0].to_dict()
    else:
        player = df.sample(1, random_state=11).iloc[0].to_dict()

    position = st.sidebar.selectbox("Position", ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"], index=["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"].index(player["position"]))
    dominant_foot = st.sidebar.selectbox("Dominant foot", ["Right", "Left", "Both"], index=["Right", "Left", "Both"].index(player["dominant_foot"]))
    nationality_region = st.sidebar.selectbox("Region", ["EU", "South America", "Africa", "UK/Ireland", "North America", "Asia"], index=["EU", "South America", "Africa", "UK/Ireland", "North America", "Asia"].index(player["nationality_region"]))

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        age = st.slider("Age", 16, 39, int(player["age"]))
        overall = st.slider("Overall", 43.0, 96.0, float(player["overall_rating"]), 0.5)
        minutes = st.slider("Minutes", 80, 4050, int(player["minutes_last_season"]), 50)
        goals = st.slider("Goals", 0, 55, int(player["goals_last_season"]))
        fitness = st.slider("Fitness", 38.0, 99.0, float(player["fitness_score"]), 0.5)
        contract = st.slider("Contract years", 0.1, 5.7, float(player["contract_years_left"]), 0.1)
        league = st.slider("League strength", 45, 100, int(player["league_strength"]))
    with col_b:
        height = st.slider("Height cm", 160, 205, int(player["height_cm"]))
        potential = st.slider("Potential", 48.0, 99.0, float(player["potential_rating"]), 0.5)
        assists = st.slider("Assists", 0, 35, int(player["assists_last_season"]))
        injury_days = st.slider("Injury days", 0, 210, int(player["injury_days_last_12m"]))
        missed = st.slider("Matches missed", 0, 38, int(player["matches_missed_last_12m"]))
        club = st.slider("Club reputation", 35.0, 100.0, float(player["club_reputation"]), 0.5)
        caps = st.slider("International caps", 0, 130, int(player["international_caps"]))

    advanced = st.sidebar.expander("Advanced Market Signals", expanded=False)
    with advanced:
        xg = st.slider("xG per 90", 0.0, 0.9, float(player["xg_per90"]), 0.01)
        xa = st.slider("xA per 90", 0.0, 0.65, float(player["xa_per90"]), 0.01)
        progressive = st.slider("Progressive actions per 90", 0.0, 11.5, float(player["progressive_actions_per90"]), 0.1)
        pressures = st.slider("Pressures per 90", 0.0, 32.0, float(player["pressures_per90"]), 0.2)
        wage = st.slider("Weekly wage EUR", 800, 680000, int(player["wage_eur_week"]), 500)
        ucl = st.slider("European minutes", 0, 1300, int(player["champions_league_minutes"]), 25)
        form = st.slider("Form index", 10.0, 99.0, float(player["form_index"]), 0.5)
        sell_on = st.slider("Sell-on potential", 0.0, 100.0, float(player["sell_on_potential"]), 0.5)
        release_clause = st.slider("Release clause ratio", 1.0, 3.2, float(player["release_clause_ratio"]), 0.05)

    return {
        "position": position,
        "dominant_foot": dominant_foot,
        "nationality_region": nationality_region,
        "age": age,
        "height_cm": height,
        "overall_rating": overall,
        "potential_rating": potential,
        "minutes_last_season": minutes,
        "goals_last_season": goals,
        "assists_last_season": assists,
        "xg_per90": xg,
        "xa_per90": xa,
        "progressive_actions_per90": progressive,
        "pressures_per90": pressures,
        "fitness_score": fitness,
        "injury_days_last_12m": injury_days,
        "matches_missed_last_12m": missed,
        "contract_years_left": contract,
        "wage_eur_week": wage,
        "international_caps": caps,
        "champions_league_minutes": ucl,
        "club_reputation": club,
        "league_strength": league,
        "form_index": form,
        "sell_on_potential": sell_on,
        "release_clause_ratio": release_clause,
    }


artifact = load_artifacts()
market_df = load_market_data()
features = build_features_from_sidebar(market_df)
prediction = predict_value(artifact, features)
low, high = uncertainty_band(prediction, features)
scenario = st.sidebar.radio("Forecast scenario", ["Prime Stability", "Breakout", "Injury Risk", "Late Career"])
forecast_years = st.sidebar.slider("Forecast horizon", 1, 5, 3)
real_player_count = int((market_df.get("data_source", pd.Series(dtype=str)) == "real_transfermarkt_top50").sum())

st.title("Football Player Market Value Prediction System")
st.caption("A scouting desk for pricing players, testing transfer scenarios, and forecasting future market value.")

top_line = st.container()
with top_line:
    st.markdown(
        f"""
        <div class="market-band">
            <span class="tag">{features["position"]}</span>
            <span class="tag">Age {features["age"]}</span>
            <span class="tag">Overall {features["overall_rating"]:.1f}</span>
            <span class="tag">Contract {features["contract_years_left"]:.1f}Y</span>
            <span class="tag">{features["nationality_region"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-label">Predicted Value</div><div class="metric-number">{eur(prediction)}</div></div>
        <div class="metric-card"><div class="metric-label">Expected Range</div><div class="metric-number">{eur(low)} to {eur(high)}</div></div>
        <div class="metric-card"><div class="metric-label">Model R2</div><div class="metric-number">{artifact["metrics"]["r2_log_value"]}</div></div>
        <div class="metric-card"><div class="metric-label">Training Rows</div><div class="metric-number">{artifact["metrics"]["rows"]:,}</div></div>
        <div class="metric-card"><div class="metric-label">Real Players</div><div class="metric-number">{real_player_count}</div></div>
        <div class="metric-card"><div class="metric-label">Dataset Median</div><div class="metric-number">{eur(artifact["metrics"]["median_market_value_eur"])}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Market Board", "Prediction Desk", "Future Value", "Data Room"])

with tabs[0]:
    filters = st.columns([1, 1, 1, 1])
    with filters[0]:
        selected_positions = st.multiselect("Positions", ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"], default=["ST", "W", "AM", "CM"])
    with filters[1]:
        max_age = st.slider("Max age", 16, 39, 27)
    with filters[2]:
        min_value = st.slider("Minimum value EUR M", 0.0, 120.0, 5.0, 0.5)
    with filters[3]:
        min_league = st.slider("Minimum league strength", 45, 100, 65)

    filtered = market_df[
        (market_df["position"].isin(selected_positions))
        & (market_df["age"] <= max_age)
        & (market_df[TARGET] >= min_value * 1_000_000)
        & (market_df["league_strength"] >= min_league)
    ].copy()
    filtered["market_value"] = filtered[TARGET].map(eur)
    filtered["weekly_wage"] = filtered["wage_eur_week"].map(eur)
    if "data_source" in filtered.columns:
        filtered["source"] = filtered["data_source"].map(
            {
                "real_transfermarkt_top50": "Real TM top 50",
                "synthetic_training_sample": "Synthetic",
            }
        ).fillna(filtered["data_source"])
    else:
        filtered["source"] = "Unknown"

    board_cols = st.columns([1.2, 1])
    with board_cols[0]:
        st.subheader("Transfer Shortlist")
        st.dataframe(
            filtered[
                [
                    "player_name",
                    "club",
                    "position",
                    "age",
                    "overall_rating",
                    "potential_rating",
                    "form_index",
                    "contract_years_left",
                    "market_value",
                    "weekly_wage",
                    "source",
                ]
            ].head(28),
            use_container_width=True,
            hide_index=True,
        )
    with board_cols[1]:
        st.subheader("Value By Age")
        if filtered.empty:
            st.info("No players match these filters.")
        else:
            chart = (
                alt.Chart(filtered.head(600))
                .mark_circle(size=78, opacity=0.72)
                .encode(
                    x=alt.X("age:Q", title="Age"),
                    y=alt.Y(f"{TARGET}:Q", title="Market value EUR", scale=alt.Scale(type="log")),
                    color=alt.Color("position:N", title="Position"),
                    size=alt.Size("potential_rating:Q", title="Potential", legend=None),
                    tooltip=["player_name", "club", "position", "age", "overall_rating", "potential_rating", TARGET],
                )
                .properties(height=430)
            )
            st.altair_chart(chart, use_container_width=True)

with tabs[1]:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Current Valuation")
        profile = pd.DataFrame(
            [
                ["Age", features["age"]],
                ["Position", features["position"]],
                ["Overall", f"{features['overall_rating']:.1f}"],
                ["Potential", f"{features['potential_rating']:.1f}"],
                ["Fitness", f"{features['fitness_score']:.1f}"],
                ["Injury days", features["injury_days_last_12m"]],
                ["Contract years", f"{features['contract_years_left']:.1f}"],
                ["League strength", features["league_strength"]],
            ],
            columns=["Signal", "Value"],
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Comparable Players")
        comp = market_df.copy()
        comp["distance"] = (
            (comp["age"] - features["age"]).abs() * 0.9
            + (comp["overall_rating"] - features["overall_rating"]).abs() * 1.6
            + (comp["potential_rating"] - features["potential_rating"]).abs() * 1.2
            + np.where(comp["position"] == features["position"], 0, 6)
            + (comp["league_strength"] - features["league_strength"]).abs() * 0.12
        )
        comp = comp.sort_values("distance").head(8).copy()
        comp["market_value"] = comp[TARGET].map(eur)
        st.dataframe(
            comp[["player_name", "club", "position", "age", "overall_rating", "potential_rating", "market_value"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Negotiation Levers")
    levers = pd.DataFrame(
        [
            {"Lever": "Short contract discount", "Impact": max(0, 2.0 - features["contract_years_left"]) * 7.5},
            {"Lever": "Injury risk discount", "Impact": min(features["injury_days_last_12m"] / 3.2, 28)},
            {"Lever": "Youth premium", "Impact": max(0, 23 - features["age"]) * 4.1},
            {"Lever": "Elite league premium", "Impact": max(0, features["league_strength"] - 75) * 0.9},
            {"Lever": "Form premium", "Impact": max(0, features["form_index"] - 70) * 0.85},
        ]
    )
    lever_chart = (
        alt.Chart(levers)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("Lever:N", sort="-x", title=None),
            x=alt.X("Impact:Q", title="Indicative impact index"),
            color=alt.Color("Lever:N", legend=None),
            tooltip=["Lever", alt.Tooltip("Impact:Q", format=".1f")],
        )
        .properties(height=250)
    )
    st.altair_chart(lever_chart, use_container_width=True)

with tabs[2]:
    forecast = future_frame(artifact, features, forecast_years, scenario)
    base_value = forecast.iloc[0]["market_value_eur"]
    forecast["value_label"] = forecast["market_value_eur"].map(eur)
    forecast["change_pct"] = ((forecast["market_value_eur"] / base_value) - 1) * 100

    cols = st.columns([1.15, 0.85])
    with cols[0]:
        st.subheader(f"{scenario} Forecast")
        line = (
            alt.Chart(forecast)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("year:N", title="Horizon"),
                y=alt.Y("market_value_eur:Q", title="Market value EUR"),
                tooltip=["year", "value_label", alt.Tooltip("change_pct:Q", format=".1f")],
            )
            .properties(height=380)
        )
        st.altair_chart(line, use_container_width=True)
    with cols[1]:
        st.subheader("Forecast Table")
        table = forecast[["year", "age", "overall_rating", "fitness_score", "contract_years_left", "value_label", "change_pct"]].copy()
        table["change_pct"] = table["change_pct"].map(lambda x: f"{x:+.1f}%")
        st.dataframe(table, use_container_width=True, hide_index=True)

    scenario_rows = []
    for item in ["Prime Stability", "Breakout", "Injury Risk", "Late Career"]:
        projection = future_frame(artifact, features, forecast_years, item)
        scenario_rows.append(
            {
                "Scenario": item,
                "Today": prediction,
                f"+{forecast_years}Y": projection.iloc[-1]["market_value_eur"],
                "Change": (projection.iloc[-1]["market_value_eur"] / prediction - 1) * 100,
            }
        )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df["Today"] = scenario_df["Today"].map(eur)
    scenario_df[f"+{forecast_years}Y"] = scenario_df[f"+{forecast_years}Y"].map(eur)
    scenario_df["Change"] = scenario_df["Change"].map(lambda x: f"{x:+.1f}%")
    st.subheader("Scenario Comparison")
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Training Dataset")
    st.write(
        "The bundled dataset now combines a current Transfermarkt top-50 real-player market-value snapshot with a larger reproducible Transfermarkt/Sofifa-style synthetic training sample. Synthetic rows provide enough breadth for regression; real rows let the market board include actual players and listed market values."
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows", f"{len(market_df):,}")
    summary_cols[1].metric("Real Players", real_player_count)
    summary_cols[2].metric("Features", len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    summary_cols[3].metric("MAE", eur(artifact["metrics"]["mae_eur"]))

    if "source_url" in market_df.columns:
        st.markdown("[Transfermarkt source table](https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop)")

    upload = st.file_uploader("Upload player CSV for batch valuation", type=["csv"])
    if upload is not None:
        uploaded = pd.read_csv(upload)
        missing = [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES if col not in uploaded.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scored = uploaded.copy()
            scored["predicted_market_value_eur"] = np.expm1(artifact["pipeline"].predict(scored[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))
            scored["predicted_value"] = scored["predicted_market_value_eur"].map(eur)
            st.dataframe(scored.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Download valuations",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="football_market_value_predictions.csv",
                mime="text/csv",
            )
    else:
        preview = market_df.head(20).copy()
        preview["market_value"] = preview[TARGET].map(eur)
        if "data_source" in preview.columns:
            preview["source"] = preview["data_source"].map(
                {
                    "real_transfermarkt_top50": "Real TM top 50",
                    "synthetic_training_sample": "Synthetic",
                }
            ).fillna(preview["data_source"])
        else:
            preview["source"] = "Unknown"
        st.dataframe(
            preview[
                [
                    "player_name",
                    "club",
                    "position",
                    "age",
                    "overall_rating",
                    "potential_rating",
                    "fitness_score",
                    "injury_days_last_12m",
                    "contract_years_left",
                    "market_value",
                    "source",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

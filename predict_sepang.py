# predict_sepang.py: main script for the 2026 Bahrain Grand Prix at Sepang.
# Combines pace, DNF risk, tyres and pit stops, then runs 10,000 simulated races.
#
# To run it for the weekend:
#   1. python sepang_tyres.py       (after each practice session)
#   2. python sepang_strategies.py
#   3. python sepang_weekend.py     (after qualifying, otherwise it uses a stand-in)
#   4. Check LINEUP, the grid penalties, COMPROMISED_QUALI, UPGRADE_FLAG, PU_SPEC_ADUO2
#   5. python predict_sepang.py

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dnf_risk
import pace_model
import race_sim
import sepang_strategies

warnings.filterwarnings("ignore")

N_SIMS = 10_000
LAPS = 56
PIT_LOSS = 22.5           # Pirelli's estimate for Sepang
# Share of 2026 races with a safety car so far (8 of 15). Sepang's last races were
# 2017 with different cars and race control, so this season is the better guide.
P_SAFETY_CAR = 8 / 15
BASE_LAP = 105.0          # a typical race lap here, from the FP2 long runs


# ════════════════════════════════════════════════════════════════════════════
# RACE WEEKEND INPUTS
# ════════════════════════════════════════════════════════════════════════════

# Same 22 as Baku. Checked against the FP1/FP2 entry list.
LINEUP = {
    "RUS": "Mercedes", "ANT": "Mercedes",
    "VER": "Red Bull Racing", "HAD": "Red Bull Racing",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "NOR": "McLaren", "PIA": "McLaren",
    "ALB": "Williams", "SAI": "Williams",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "BEA": "Haas F1 Team", "OCO": "Haas F1 Team",
    "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "HUL": "Audi", "BOR": "Audi",
    "GAS": "Alpine", "COL": "Alpine",
    "PER": "Cadillac", "BOT": "Cadillac",
}

# Qualifying, rebuilt from the session lap data. The grid comes from the
# classification and not from sorting the times, because everyone who reached Q3
# starts ahead of everyone knocked out in Q2 even if the Q2 car was quicker.
_Q = pd.read_csv("data/sepang2026_qualifying.csv")
STAND_IN = _Q["status"].str.contains("STAND-IN").any()
QUALI_TIME = dict(zip(_Q.driver, _Q.time_s))
QUALI_SPEED_TRAP = dict(zip(_Q.driver, _Q.speed_trap))
GRID = dict(zip(_Q.driver, _Q.P))

# Grid penalties announced on Friday (F1.com, "What the teams said"):
#   HAD  5 places, new ICE
#   COL  15 places: 5 for the Baku collision, 10 for a new ICE
#   LIN  back of the grid, several new power unit parts
GRID_PENALTY_PLACES = {"HAD": 5, "COL": 15}
GRID_PENALTY_BACK = {"LIN"}

# Drivers whose qualifying lap says nothing useful about their car, because they
# crashed or hit a problem. Their lap gets swapped for their practice pace, but they
# still start where they qualified. Fill in after qualifying.
COMPROMISED_QUALI = set()

# Teams whose season-long form describes an older car, so lean on this weekend's pace
# instead. Williams and Audi upgraded at Baku (one race of data since), Mercedes
# brought "a significant aerodynamic update" here (The Race, Shovlin on Friday).
UPGRADE_FLAG = {"Williams": True, "Audi": True, "Mercedes": True}

# Ferrari drivers whose engine is newer than the one they QUALIFIED with. Empty here:
# HAM and LEC both qualified on ADUO2 (Ferrari's engine since Monza), so their lap
# already includes its ~0.2s. Adding it again would count the engine twice, which
# the Baku version of this model did.
PU_SPEC_ADUO2 = set()


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD THE CAR TABLE
# ════════════════════════════════════════════════════════════════════════════

def merit_grid():
    """Qualifying order straight from the lap times, before any penalty."""
    return {d: i + 1 for i, d in enumerate(sorted(LINEUP, key=lambda d: QUALI_TIME[d]))}


def race_grid():
    """Where everyone actually starts. A place penalty drops you that many slots and
    you go behind whoever is already there; back-of-grid drivers go last of all, in
    their qualifying order."""
    grid = dict(GRID) if GRID else merit_grid()
    key = {d: grid[d] + GRID_PENALTY_PLACES.get(d, 0) + (0.5 if d in GRID_PENALTY_PLACES else 0)
           for d in grid}
    for d in GRID_PENALTY_BACK:
        key[d] = 1000 + grid[d]
    return {d: i + 1 for i, d in enumerate(sorted(grid, key=key.get))}


def build_cars():
    lineup_df = pd.DataFrame({"driver": list(LINEUP), "team": list(LINEUP.values())})
    lineup_df["supplier"] = lineup_df["team"].map(dnf_risk.SUPPLIER)

    # DNF risk, trained on this season
    events = dnf_risk.build_events(dnf_risk.load_results())
    cars = dnf_risk.car_probabilities(events, lineup_df.assign(category=np.nan), k=5)
    cars["p_tyre"] = dnf_risk.P_TYRE_FAILURE
    cars["p_dnf"] = 1 - (1 - cars.p_dnf) * (1 - cars.p_tyre)

    # Pace, from the ensemble trained on the season so far
    raw = pd.read_csv("data/pace_dataset_2026.csv")
    feats = pace_model.make_features(raw)
    quali = pd.DataFrame({
        "driver": list(LINEUP),
        "quali_gap_pct": [100 * (QUALI_TIME[d] - min(QUALI_TIME.values())) / min(QUALI_TIME.values())
                          for d in LINEUP],
        "quali_speed_trap": [QUALI_SPEED_TRAP[d] for d in LINEUP],
    })
    # A crashed qualifying lap tells you nothing about the car, so swap it for that
    # driver's practice pace. Season form is the fallback if there's no practice data.
    if COMPROMISED_QUALI:
        practice = pd.read_csv("data/sepang2026_practice.csv", index_col=0)["best_practice_gap_pct"]
        season = raw.groupby("driver")["quali_gap_pct"].median()
        is_bad = quali["driver"].isin(COMPROMISED_QUALI)
        sub = quali.loc[is_bad, "driver"].map(practice)
        sub = sub.fillna(quali.loc[is_bad, "driver"].map(season)).fillna(quali["quali_gap_pct"].median())
        quali.loc[is_bad, "quali_gap_pct"] = sub
        quali.loc[is_bad, "quali_speed_trap"] = quali["quali_speed_trap"].median()

    quali["speed_rel"] = quali["quali_speed_trap"] - quali["quali_speed_trap"].median()
    # The grid feature here is ranked from pace, not from the real starting grid. A
    # penalty or a crash changes where you start, not how quick the car is.
    quali["grid"] = quali["quali_gap_pct"].rank(method="first")
    team_form = raw.groupby("team")["pace_gap_pct"].mean()
    quali["team_form"] = quali["driver"].map(LINEUP).map(team_form).fillna(raw["pace_gap_pct"].mean())
    supplier_speed = feats.groupby("supplier")["speed_rel"].mean() if "speed_rel" in feats else pd.Series(dtype=float)
    quali["supplier_speed"] = quali["driver"].map(LINEUP).map(dnf_risk.SUPPLIER).map(supplier_speed).fillna(0)

    tr = feats.dropna(subset=["pace_gap_pct"])
    ms = [m for name, m in pace_model.models().items() if name[0] in "234"]
    quali["pace_gap_pct"] = np.mean(
        [m.fit(tr[pace_model.FEATURES], tr["pace_gap_pct"]).predict(quali[pace_model.FEATURES]) for m in ms], axis=0)

    cars = cars.merge(quali[["driver", "pace_gap_pct", "quali_gap_pct"]], on="driver")
    # Only now does the real starting grid come in, penalties and all
    cars["grid"] = cars["driver"].map(race_grid())

    # No track specialist nudge here: nobody has raced these cars at Sepang

    # ADUO2 engine, roughly 0.2s a lap
    cars.loc[cars["driver"].isin(PU_SPEC_ADUO2), "pace_gap_pct"] -= 100 * 0.2 / BASE_LAP

    # For upgraded teams the season form describes the old car, so blend it halfway
    # towards what they actually did in qualifying
    upgraded = cars["team"].map(UPGRADE_FLAG).fillna(False)
    cars.loc[upgraded, "pace_gap_pct"] = (
        0.5 * cars.loc[upgraded, "pace_gap_pct"] + 0.5 * cars.loc[upgraded, "quali_gap_pct"])

    # Erratic drivers get a wider spread of outcomes
    cars["noise_mult"] = dnf_risk.variance_multiplier(cars["p_incident"])

    # How kind each car is to its tyres, from the practice long runs (sepang_tyres.py)
    team_wear = pd.read_csv("data/sepang_team_wear.csv", index_col="team")["deg_mult"]
    cars["deg_mult"] = cars["team"].map(team_wear).fillna(1.0)

    return cars.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: RUN THE MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════

def run(cars, params=None):
    """Every race in one call so each one gets its own roll of form, retirements,
    strategy and safety car. Each car picks its plan using its own tyre wear, so a
    car that's gentle on its tyres leans towards fewer stops.
    Returns (finish, retired, stops each car made in each race)."""
    plans = pd.read_pickle("data/sepang_strategies.pkl")
    tyres = pd.read_csv("data/sepang_tyres.csv", index_col="compound")
    wear, offset = tyres["wear"].to_dict(), tyres["offset"].to_dict()
    probs = sepang_strategies.plan_probabilities(plans, wear, offset, cars["deg_mult"].values)
    p = dict(base_lap=BASE_LAP, pit_loss=PIT_LOSS, deg=wear, offset=offset, p_safety_car=P_SAFETY_CAR, **(params or {}))
    return race_sim.simulate(cars, LAPS, n=N_SIMS, seed=42, params=p, strategies=plans,
                             plan_probs=probs, return_retirements=True, return_stops=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: TURN THE SIMULATIONS INTO RESULTS
# ════════════════════════════════════════════════════════════════════════════

def summarize(cars, finish, retired):
    """Counts everything from the simulated races, so the DNF column and the
    finishing positions always describe the same set of races."""
    cars = cars.copy()
    cars["exp_finish"] = finish.mean(0)
    finished = ~retired
    cars["win_%"] = 100 * ((finish == 1) & finished).mean(0)
    cars["top3_%"] = 100 * ((finish <= 3) & finished).mean(0)
    cars["points_%"] = 100 * ((finish <= 10) & finished).mean(0)
    cars["dnf_%"] = 100 * retired.mean(0)
    order = cars.sort_values("exp_finish").reset_index(drop=True)
    order.insert(0, "predicted_position", range(1, len(order) + 1))
    return order


def charts(order, finish, drivers):
    fig, ax = plt.subplots(figsize=(10, 7))
    top = order.head(12).iloc[::-1]
    ax.barh(top["driver"], top["win_%"], color="#c00000", label="Win %")
    ax.barh(top["driver"], top["top3_%"] - top["win_%"], left=top["win_%"], color="#e6a4a4", label="Top 3 % (extra)")
    ax.set_xlabel("Probability (%)"); ax.set_title("Bahrain GP at Sepang 2026 - win / top-3 probability"); ax.legend()
    fig.tight_layout(); fig.savefig("data/sepang_probability_chart.png", dpi=150)
    plt.close(fig)

    # Heatmap: how often each driver lands in each position
    C = finish.shape[1]
    heat = np.zeros((C, C))
    for j in range(len(drivers)):
        counts = np.bincount(finish[:, j], minlength=C + 1)[1:]
        heat[j] = counts / finish.shape[0]
    row_order = order["driver"].map({d: i for i, d in enumerate(drivers)}).values
    heat = heat[row_order]

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    im = ax2.imshow(heat, aspect="auto", cmap="Reds")
    ax2.set_yticks(range(C)); ax2.set_yticklabels(order["driver"])
    ax2.set_xticks(range(C)); ax2.set_xticklabels([str(p) for p in range(1, C + 1)])
    ax2.set_xlabel("Finishing position"); ax2.set_title("Bahrain GP at Sepang 2026 - finishing position probability")
    fig2.colorbar(im, ax=ax2, label="Probability")
    fig2.tight_layout(); fig2.savefig("data/sepang_heatmap.png", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    cars = build_cars()
    finish, retired, stops = run(cars)
    order = summarize(cars, finish, retired)
    order["avg_stops"] = order["driver"].map(dict(zip(cars["driver"], stops.mean(0)))).round(2)
    charts(order, finish, cars["driver"].tolist())
    pd.DataFrame(finish, columns=cars["driver"]).to_csv("data/sepang_sim_raw.csv", index=False)
    order.to_csv("data/sepang_prediction.csv", index=False)

    print("=" * 70)
    print("  BAHRAIN GP AT SEPANG 2026 - PREDICTED RESULT  (if it stays dry)")
    if STAND_IN:
        print("  ** STAND-IN GRID FROM PRACTICE: qualifying hasn't happened yet **")
    print(f"  {len(_Q)} drivers | {N_SIMS:,} simulated races")
    print("=" * 70)
    cols = ["predicted_position", "driver", "team", "grid", "win_%", "top3_%", "points_%", "dnf_%", "avg_stops"]
    print(order[cols].round(1).to_string(index=False))
    if GRID_PENALTY_PLACES:
        print("\n  Place penalties: " + ", ".join(f"{d} -{n}" for d, n in GRID_PENALTY_PLACES.items()))
    if GRID_PENALTY_BACK:
        print(f"  Back of the grid: {', '.join(sorted(GRID_PENALTY_BACK))}")
    if COMPROMISED_QUALI:
        print(f"  Qualifying lap treated as unrepresentative: {', '.join(sorted(COMPROMISED_QUALI))}")

    # ════════════════════════════════════════════════════════════════════════
    # THE MODEL'S THREE CALLS (Lynda vs the model)
    # ════════════════════════════════════════════════════════════════════════
    # Winner: most race wins. Podium: the three most often in the top 3.
    # Stops: how many stops the winning car made, counted across every race it won.
    win = order.sort_values("win_%", ascending=False).iloc[0]
    podium = order.sort_values("top3_%", ascending=False).head(3)
    winner_stops = stops[(finish == 1) & ~retired]
    mix = pd.Series(winner_stops).value_counts(normalize=True).sort_index()
    print("\n" + "=" * 70)
    print("  THE MODEL'S CALLS")
    print("=" * 70)
    print(f"  1. Winner:   {win['driver']}  ({win['win_%']:.0f}% of races)")
    print(f"  2. Podium:   " + ", ".join(f"{d} ({p:.0f}%)" for d, p in zip(podium["driver"], podium["top3_%"])))
    print(f"  3. Stops for the winner:  {mix.idxmax()}  ("
          + ", ".join(f"{k} stops {v:.0%}" for k, v in mix.items()) + ")")

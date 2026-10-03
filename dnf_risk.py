# dnf_risk.py: works out how likely each car is to retire, and from what.

import warnings
import numpy as np
import pandas as pd
import fastf1
import pace_data

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD EVERY RACE RESULT
# ════════════════════════════════════════════════════════════════════════════
# One row per driver per race. Teams are read per round, since lineups change
# mid-season and you can't just map a driver to one team for the whole year.

# Past races at the track being predicted. Baku had five recent ones; Sepang's last
# race was 2017 with completely different cars, so there's nothing comparable to add.
TRACK_HISTORY = []          # e.g. [(y, "Azerbaijan", f"{y}-Baku") for y in range(2021, 2026)]
TRACK_HISTORY_FILE = None   # e.g. "data/baku_race_retirements_2021_2025.csv"

# No tyre has failed in a 2026 race (0 in about 330 starts), but Sepang is the
# roughest surface of the year. ASSUMPTION: 0.5%, a quarter of Baku's 2%.
P_TYRE_FAILURE = 0.5 / 100


def load_results():
    rows = []
    races = [(2026, r, f"2026-R{r}") for r in range(1, pace_data.LAST_ROUND + 1)] + TRACK_HISTORY
    for year, rnd, label in races:
        s = fastf1.get_session(year, rnd, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        for r in s.results.itertuples():
            rows.append(dict(race=label, year=year, driver=r.Abbreviation,
                             team=r.TeamName, status=r.Status))
    return pd.DataFrame(rows)


# Which engine each team runs. Engine failures get scored per supplier, since
# four teams sharing a Mercedes are all evidence about the same engine.
SUPPLIER = {
    "Mercedes": "Mercedes", "McLaren": "Mercedes", "Williams": "Mercedes", "Alpine": "Mercedes",
    "Ferrari": "Ferrari", "Haas F1 Team": "Ferrari", "Cadillac": "Ferrari",
    "Red Bull Racing": "Red Bull-Ford", "Racing Bulls": "Red Bull-Ford",
    "Aston Martin": "Honda", "Audi": "Audi",
}


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: ATTACH THE REASON FOR EVERY RETIREMENT
# ════════════════════════════════════════════════════════════════════════════
# FastF1 tells you who retired but almost never why, so the causes come from the
# two CSVs in data/. Categories: power_unit, mechanical, crash, contact,
# tyre_failure, unknown, dns

def build_events(results):
    r26 = pd.read_csv("data/retirements_2026.csv")
    r26["race"] = "2026-R" + r26["round"].astype(str)
    ev = r26[["race", "driver", "category"]]
    if TRACK_HISTORY_FILE:
        hist = pd.read_csv(TRACK_HISTORY_FILE)
        hist["race"] = hist["year"].astype(str) + "-Baku"
        ev = pd.concat([ev, hist[["race", "driver", "category"]]])
    df = results.merge(ev, on=["race", "driver"], how="left")
    df["supplier"] = df["team"].map(SUPPLIER)
    # If you never started you can't fail during the race, so drop those rows
    df = df[df["category"] != "dns"].copy()
    # Retirements with no known cause get counted as mechanical
    df["category"] = df["category"].replace("unknown", "mechanical")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: SHRINKAGE
# ════════════════════════════════════════════════════════════════════════════
# rate = (events + k * grid average) / (starts + k)
# k is basically how many starts worth of doubt to add. Small samples get pulled
# towards the grid average, big ones get trusted. Never gives you 0% or 100%.

def shrunk_rates(df, group_col, causes, k):
    hits = df["category"].isin(causes)
    prior = hits.mean()
    g = df.assign(hit=hits).groupby(group_col)["hit"].agg(["sum", "count"])
    return (g["sum"] + k * prior) / (g["count"] + k)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: ONE DNF PROBABILITY PER CAR
# ════════════════════════════════════════════════════════════════════════════
# Treat the causes as independent, so P(finishes) is just all of them multiplied.
# Engine goes by supplier, mechanical by team, crashes by driver.

def car_probabilities(train, cars, k):
    pu = shrunk_rates(train, "supplier", ["power_unit"], k)
    mech = shrunk_rates(train, "team", ["mechanical"], k)
    inc = shrunk_rates(train, "driver", ["crash", "contact"], k)
    prior = lambda causes: train["category"].isin(causes).mean()
    out = cars.copy()
    out["p_pu"] = out["supplier"].map(pu).fillna(prior(["power_unit"]))
    out["p_mech"] = out["team"].map(mech).fillna(prior(["mechanical"]))
    out["p_incident"] = out["driver"].map(inc).fillna(prior(["crash", "contact"]))
    out["p_dnf"] = 1 - (1 - out.p_pu) * (1 - out.p_mech) * (1 - out.p_incident)
    return out


# ════════════════════════════════════════════════════════════════════════════
# DRIVER VARIANCE
# ════════════════════════════════════════════════════════════════════════════
# Erratic drivers should get a wider spread of outcomes, not just a higher crash
# number. This scales how much their race swings, reusing p_incident rather than
# inventing a second metric for the same thing.

def variance_multiplier(p_incident, lo=0.7, hi=1.6):
    avg = p_incident.mean()
    mult = 0.7 + (p_incident / avg) * 0.5
    return mult.clip(lo, hi)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: PICK k BY TESTING IT ON PAST RACES
# ════════════════════════════════════════════════════════════════════════════
# Train on earlier rounds only, predict the next one, score with log loss.
# Lower is better, and k=1e9 just means "always use the grid average".

def choose_k(df26, ks):
    rounds = sorted(df26["race"].unique(), key=lambda s: int(s.split("R")[1]))
    scores = {k: [] for k in ks}
    for i in range(5, len(rounds)):
        train, test = df26[df26.race.isin(rounds[:i])], df26[df26.race == rounds[i]]
        actual = (test["category"].notna() & (test["category"] != "tyre_failure")).astype(float).values
        for k in ks:
            p = car_probabilities(train, test, k)["p_dnf"].clip(0.005, 0.95).values
            scores[k].append(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
    return pd.Series({k: np.mean(v) for k, v in scores.items()})


if __name__ == "__main__":
    df = build_events(load_results())
    df26 = df[df.year == 2026]

    ll = choose_k(df26, [1, 5, 10, 20, 30, 50, 100, 1e9])
    print("Log loss by k (lower = better; 1e9 = grid average only)")
    print(ll.round(4).to_string())
    best_k = ll.idxmin()
    print(f"\nBest k = {best_k}")

    # Use the latest lineup
    lineup = df26[df26.race == f"2026-R{pace_data.LAST_ROUND}"][["driver", "team", "supplier"]]
    probs = car_probabilities(df, lineup, best_k)
    probs["p_tyre"] = P_TYRE_FAILURE
    probs["p_dnf"] = 1 - (1 - probs.p_dnf) * (1 - probs.p_tyre)
    print("\nRace DNF probability per car")
    print((probs.set_index("driver")[["team", "p_pu", "p_mech", "p_incident", "p_dnf"]]
           .sort_values("p_dnf", ascending=False)).round(3).to_string())
    probs.to_csv("data/dnf_probabilities.csv", index=False)

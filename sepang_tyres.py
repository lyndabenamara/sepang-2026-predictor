# sepang_tyres.py: tyre wear and compound speed for Sepang, from this weekend's practice
# There's no race data from Sepang with these cars, so wear is measured from the long
# runs in practice. Re-run it after each practice session: it uses every one so far.
# Produces: data/sepang_tyres.csv (wear and speed per compound) and
#           data/sepang_team_wear.csv (how kind each car is to its tyres)

import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

ROUND = 16
SESSIONS = ["FP1", "FP2", "FP3"]
MIN_RUN = 6            # laps in a row on one set before it counts as a long run
SLOW_LAP = 1.2         # seconds above the stint's median that marks traffic or a cool down lap

# Measured at Spa and Shanghai 2026 (both -0.058 s per lap of fuel burned), scaled
# down for Sepang's shorter lap. In practice we can't separate fuel from wear, because
# both change by the same amount every lap, so this gets added back on afterwards.
FUEL_GAIN = 0.058 * 5.543 / 7.004 * 0.5 + 0.058 * 5.543 / 5.451 * 0.5

# How much slower a fresh set is than a fresh soft.
#   MEDIUM: Pirelli after Friday at Sepang, "around half a second"
#   HARD:   medium + 0.45s, where Spa (0.43), Shanghai (0.52) and F1's Shanghai
#           graphic (0.47) all agree
OFFSET = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 0.95}

# Nobody has run the hard here. Spa used the same three compounds (C2/C3/C4), and
# there the hard wore at this fraction of the medium's rate. Shanghai said 0.23,
# so that's the obvious "what if" to test.
HARD_VS_MEDIUM_WEAR = 0.68

# How much of a team's practice wear difference to believe. tyre_backtest.py checked
# this on every 2026 race: practice wear and race wear by team had a correlation of
# 0.04 and the best fit was 0.00. Practice differences are mostly fuel loads and run
# plans, not the car, so every team gets the field average.
TEAM_TRUST = 0.0


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: FIND THE LONG RUNS
# ════════════════════════════════════════════════════════════════════════════
# A long run is one set of tyres driven for several laps in a row. Pit laps go,
# and so does anything well off the stint's typical time (traffic, cool down laps,
# mistakes). That's done twice, since one very slow lap drags the median up.

def long_runs(rnd=ROUND, sessions=SESSIONS):
    runs = []
    for name in sessions:
        try:
            s = fastf1.get_session(2026, rnd, name)
            s.load(telemetry=False, weather=False, messages=False)
            laps = s.laps.copy()
        except Exception:
            continue                  # session hasn't happened yet
        if laps.empty:
            continue
        laps["t"] = laps["LapTime"].dt.total_seconds()
        for (driver, stint), g in laps.groupby(["Driver", "Stint"]):
            g = g[g["PitInTime"].isna() & g["PitOutTime"].isna() & g["t"].notna()]
            g = g[g["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
            for _ in range(2):
                g = g[g["t"] < g["t"].median() + SLOW_LAP]
            if len(g) >= MIN_RUN:
                runs.append(g.assign(run=f"{name}-{driver}-{int(stint)}", session=name))
    return pd.concat(runs, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: WEAR RATE
# ════════════════════════════════════════════════════════════════════════════
# lap_time = run_baseline + wear * tyre_age
# Every run gets its own baseline, so different fuel loads and engine modes between
# runs don't matter: only how lap time changes WITHIN a run is used.
# Soft and medium share one rate, because Pirelli found they "displayed very similar
# levels of performance in terms of both pace and degradation" and our data agrees.

def fit_wear(runs, group=None):
    X = pd.get_dummies(runs["run"]).astype(float)
    age = runs["TyreLife"].astype(float).values
    names = sorted(runs[group].unique()) if group else ["all"]
    for g in names:
        X[f"wear_{g}"] = np.where(runs[group].values == g, age, 0.0) if group else age
    coef, *_ = np.linalg.lstsq(X.values, runs["t"].values, rcond=None)
    resid = runs["t"].values - X.values @ coef
    # Standard errors, so we know how much to trust each team's number
    sigma2 = resid.var() * len(resid) / (len(resid) - X.shape[1])
    cov = sigma2 * np.linalg.pinv(X.values.T @ X.values)
    k = len(names)
    return pd.DataFrame({"wear": coef[-k:], "se": np.sqrt(np.diag(cov)[-k:])}, index=names)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: TEAM BY TEAM, PULLED TOWARDS THE AVERAGE
# ════════════════════════════════════════════════════════════════════════════
# Some cars really are kinder to their tyres, but a team with 3 runs in practice
# could also just have had an easy programme. So each team's number is pulled
# towards the field average, more strongly the noisier it is:
#   trust = real spread between teams / (real spread + this team's noise)
# The real spread is the spread we see minus the part that's just noise.

def team_wear(runs, overall):
    t = fit_wear(runs, group="Team")
    spread = max(float(t["wear"].var()) - float((t["se"] ** 2).mean()), 0.0)
    t["trust"] = spread / (spread + t["se"] ** 2)
    t["wear_shrunk"] = overall + t["trust"] * (t["wear"] - overall)
    t["runs"] = runs.groupby("Team")["run"].nunique()
    t["deg_mult"] = 1 + TEAM_TRUST * (t["wear_shrunk"] / overall - 1)
    return t


if __name__ == "__main__":
    runs = long_runs()
    print(f"{runs['run'].nunique()} long runs, {len(runs)} laps, from "
          f"{', '.join(sorted(runs['session'].unique()))}")
    print("compounds:", runs.groupby("Compound")["run"].nunique().to_dict())

    by_compound = fit_wear(runs, group="Compound")
    print("\nWear by compound, as seen (s per lap of tyre age):")
    print(by_compound.round(3).to_string())

    seen = fit_wear(runs).loc["all"]
    wear = seen["wear"] + FUEL_GAIN
    print(f"\nSoft and medium together: {seen['wear']:.3f} seen + {FUEL_GAIN:.3f} hidden by fuel burn "
          f"= {wear:.3f} s/lap  (Pirelli: 0.2-0.3)")

    tyres = pd.DataFrame({
        "compound": ["SOFT", "MEDIUM", "HARD"],
        "wear": [wear, wear, wear * HARD_VS_MEDIUM_WEAR],
        "offset": [OFFSET["SOFT"], OFFSET["MEDIUM"], OFFSET["HARD"]],
        "source": ["practice long runs", "practice long runs", f"practice x {HARD_VS_MEDIUM_WEAR} (Spa ratio)"],
    })
    print("\n" + tyres.round(3).to_string(index=False))
    tyres.to_csv("data/sepang_tyres.csv", index=False)

    teams = team_wear(runs, seen["wear"])
    print(f"\nWear by team (seen, then pulled towards the average; TEAM_TRUST = {TEAM_TRUST} from tyre_backtest.py):")
    print(teams.sort_values("wear_shrunk")[["runs", "wear", "se", "trust", "wear_shrunk", "deg_mult"]]
          .round(3).to_string())
    teams[["deg_mult"]].rename_axis("team").to_csv("data/sepang_team_wear.csv")

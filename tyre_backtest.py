# tyre_backtest.py: does a team's tyre wear in practice tell you its tyre wear in the race?
# The Sepang prediction leans hard on practice long runs (McLaren looked far kinder to
# its tyres than anyone). This checks that idea on every 2026 race so far: measure
# each team's wear in practice, measure it again in the race, compare.
# Produces: data/tyre_backtest.csv and the trust factor printed at the end.

import warnings
import numpy as np
import pandas as pd
import fastf1

import pace_data
import sepang_tyres

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: EACH TEAM'S WEAR IN THE RACE
# ════════════════════════════════════════════════════════════════════════════
# Same idea as practice: every stint gets its own baseline, and only how lap time
# changes within a stint counts. In a race the fuel load falls at the same rate for
# everyone, so one shared fuel term takes it out.

def race_runs(rnd):
    s = fastf1.get_session(2026, rnd, "R")
    s.load(telemetry=False, weather=False, messages=False)
    laps = s.laps
    laps = laps[(laps["TrackStatus"] == "1") & (laps["LapNumber"] > 1)]
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna() & laps["LapTime"].notna()]
    laps = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    laps["t"] = laps["LapTime"].dt.total_seconds()
    runs = []
    for (driver, stint), g in laps.groupby(["Driver", "Stint"]):
        for _ in range(2):
            g = g[g["t"] < g["t"].median() + sepang_tyres.SLOW_LAP]
        if len(g) >= sepang_tyres.MIN_RUN:
            runs.append(g.assign(run=f"R-{driver}-{int(stint)}"))
    return pd.concat(runs, ignore_index=True)


def race_team_wear(runs):
    X = pd.get_dummies(runs["run"]).astype(float)
    age = runs["TyreLife"].astype(float).values
    teams = sorted(runs["Team"].unique())
    for tm in teams:
        X[tm] = np.where(runs["Team"].values == tm, age, 0.0)
    X["lap_no"] = runs["LapNumber"].astype(float).values
    coef, *_ = np.linalg.lstsq(X.values, runs["t"].values, rcond=None)
    return pd.Series(coef[-len(teams) - 1:-1], index=teams)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPARE, ROUND BY ROUND
# ════════════════════════════════════════════════════════════════════════════
# Everything is measured as "how much better or worse than the field average this
# weekend", so a high wear track and a low wear track can sit in the same comparison.

def compare(rounds):
    rows = []
    for rnd in rounds:
        try:
            prac = sepang_tyres.long_runs(rnd=rnd)
            p = sepang_tyres.fit_wear(prac, group="Team")
            r = race_team_wear(race_runs(rnd))
        except Exception as e:
            print(f"round {rnd}: skipped ({type(e).__name__})", flush=True)
            continue
        both = p.index.intersection(r.index)
        p_dev = p.loc[both, "wear"] - p.loc[both, "wear"].mean()
        r_dev = r.loc[both] - r.loc[both].mean()
        for tm in both:
            rows.append(dict(round=rnd, team=tm, practice_dev=p_dev[tm], practice_se=p.loc[tm, "se"],
                             race_dev=r_dev[tm], practice_runs=prac.loc[prac["Team"] == tm, "run"].nunique()))
        print(f"round {rnd}: {len(both)} teams, {prac['run'].nunique()} practice long runs", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = compare(range(1, pace_data.LAST_ROUND + 1))
    d.to_csv("data/tyre_backtest.csv", index=False)

    # Trust factor: if practice says a team is 0.1 s/lap better than average, how much
    # better does it turn out in the race? 1 = fully, 0 = practice tells you nothing.
    trust = (d["practice_dev"] * d["race_dev"]).sum() / (d["practice_dev"] ** 2).sum()
    corr = d["practice_dev"].corr(d["race_dev"])
    print(f"\n{len(d)} team-weekends from {d['round'].nunique()} races")
    print(f"correlation practice vs race wear: {corr:.2f}")
    print(f"trust factor (race = trust x practice): {trust:.2f}")

    # Same thing but only where practice had a decent sample, like McLaren's 4 runs at Sepang
    big = d[d["practice_runs"] >= 3]
    t_big = (big["practice_dev"] * big["race_dev"]).sum() / (big["practice_dev"] ** 2).sum()
    print(f"teams with 3+ practice long runs: {len(big)} team-weekends, "
          f"correlation {big['practice_dev'].corr(big['race_dev']):.2f}, trust {t_big:.2f}")

    # How often does a team that looks clearly best in practice turn out better than average in the race?
    best = d.loc[d.groupby("round")["practice_dev"].idxmin()]
    print(f"\nthe team with the lowest practice wear each weekend was better than average in the race "
          f"{(best['race_dev'] < 0).sum()} of {len(best)} times")
    print(best[["round", "team", "practice_dev", "race_dev"]].round(3).to_string(index=False))

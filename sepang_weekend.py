# sepang_weekend.py: pulls this weekend's qualifying and practice into data/
# Produces: data/sepang2026_qualifying.csv and data/sepang2026_practice.csv
#
# Before qualifying has happened it writes a STAND-IN qualifying file built from
# practice, so the whole prediction can be tested. Every row is marked as such.

import warnings
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

ROUND = 16


def load(name):
    try:
        s = fastf1.get_session(2026, ROUND, name)
        # messages=True matters: without it FastF1 can leave the Q1/Q2/Q3 times empty
        s.load(telemetry=False, weather=True, messages=True)
        if s.laps.empty:
            return None
        return s
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# PRACTICE: each driver's best lap in each session, as a gap to the fastest
# ════════════════════════════════════════════════════════════════════════════

def practice_table():
    out = {}
    for name in ["FP1", "FP2", "FP3"]:
        s = load(name)
        if s is None:
            continue
        best = s.laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
        out[name] = (100 * (best - best.min()) / best.min()).round(3)
    df = pd.DataFrame(out)
    df["best_practice_gap_pct"] = df.min(axis=1)
    return df.rename_axis("driver")


# ════════════════════════════════════════════════════════════════════════════
# QUALIFYING: grid from the official classification, pace from the best lap
# ════════════════════════════════════════════════════════════════════════════
# Everyone who reached Q3 starts ahead of everyone knocked out in Q2, even if a Q2
# car set a quicker time, so the grid order comes from the classification.
# Pace is a different question: a deleted or messed-up Q3 lap says nothing about the
# car, so each driver's pace is their best lap from any part of qualifying.

def qualifying_table(q):
    res = q.results.set_index("Abbreviation")
    if res[["Q1", "Q2", "Q3"]].isna().all().all():
        raise RuntimeError("FastF1 returned no qualifying times. Wait a few minutes and run it again.")
    best = res[["Q1", "Q2", "Q3"]].min(axis=1).dt.total_seconds()
    reached = res[["Q1", "Q2", "Q3"]].notna().sum(axis=1)
    laps = q.laps[q.laps["LapTime"].notna()]
    trap = laps.loc[laps.groupby("Driver")["LapTime"].idxmin()].set_index("Driver")["SpeedST"]
    df = pd.DataFrame({"P": res["Position"], "driver": res.index, "status": "reached Q" + reached.astype(str),
                       "time_s": best, "speed_trap": res.index.map(trap)})
    if df["P"].isna().any():
        raise RuntimeError("FastF1 has no qualifying classification yet. Wait a few minutes and run it again.")
    return df.sort_values("P").astype({"P": int}).reset_index(drop=True)


def stand_in_qualifying():
    """No qualifying yet: rank everyone by their best practice lap instead."""
    laps = pd.concat([s.laps for s in (load(n) for n in ["FP1", "FP2", "FP3"]) if s is not None],
                     ignore_index=True)
    laps = laps[laps["LapTime"].notna()]
    best = laps.loc[laps.groupby("Driver")["LapTime"].idxmin()]
    df = pd.DataFrame({"driver": best["Driver"].values,
                       "time_s": best["LapTime"].dt.total_seconds().values,
                       "speed_trap": best["SpeedST"].values}).sort_values("time_s")
    df.insert(0, "P", range(1, len(df) + 1))
    df.insert(2, "status", "STAND-IN from practice")
    return df


if __name__ == "__main__":
    prac = practice_table()
    prac.to_csv("data/sepang2026_practice.csv")
    print("Practice gaps (%):")
    print(prac.sort_values("best_practice_gap_pct").to_string())

    q = load("Q")
    quali = qualifying_table(q) if q is not None else stand_in_qualifying()
    quali.to_csv("data/sepang2026_qualifying.csv", index=False)
    print("\nQualifying" + ("" if q is not None else "  ** STAND-IN, qualifying hasn't happened yet **") + ":")
    print(quali.to_string(index=False))

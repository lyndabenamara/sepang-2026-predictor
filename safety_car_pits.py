# safety_car_pits.py: who dives into the pits when a safety car comes out
# Produces: the chance a car pits under a safety car, split by whether it has
# already made a stop and how old its tyres are. The race simulator uses this so
# cars react to a safety car instead of sticking to a plan made before the start.

import warnings
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

RACES = [(2026, r) for r in range(1, 16)] + [(y, "Azerbaijan") for y in range(2021, 2026)]

# Tyre age (laps) past which a car counts as "due" a stop. Chosen by looking at the
# data, where the pit rate jumps: fresh tyres rarely come in, older ones mostly do.
DUE_AGE_FIRST_STOP = 8      # hasn't stopped yet
DUE_AGE_EXTRA_STOP = 25     # already stopped, so this would be a bonus stop


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: FIND EVERY SAFETY CAR
# ════════════════════════════════════════════════════════════════════════════
# A safety car is a run of laps where the leader's track status includes "4".
# Any period with a red flag ("5") is thrown out, because under a red flag every
# car changes tyres in the pit lane for free, which isn't a strategy decision.

def safety_car_periods(laps):
    lead = laps[laps["Position"] == 1].set_index("LapNumber")["TrackStatus"].astype(str)
    periods = []
    for lap, status in lead.items():
        if "4" not in status:
            continue
        if periods and lap == periods[-1][-1] + 1:
            periods[-1].append(int(lap))
        else:
            periods.append([int(lap)])
    red = lambda p: laps[laps["LapNumber"].between(p[0], p[-1] + 8)]["TrackStatus"].astype(str).str.contains("5").any()
    return [p for p in periods if not red(p)]


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: WHAT EACH CAR DID
# ════════════════════════════════════════════════════════════════════════════
# For every car running when the safety car came out: how old were its tyres, had
# it already stopped, and did it pit before the restart?

def reactions(laps, periods):
    rows = []
    for p in periods:
        first = p[0]
        for driver, d in laps.groupby("Driver"):
            d = d.set_index("LapNumber")
            if first - 1 not in d.index or first not in d.index:
                continue
            window = [k for k in p + [p[-1] + 1] if k in d.index]
            rows.append(dict(
                lap=first,
                tyre_age=d.loc[first - 1, "TyreLife"],
                already_stopped=d.loc[:first - 1, "PitInTime"].notna().any(),
                pitted=d.loc[window, "PitInTime"].notna().any(),
            ))
    return rows


def pit_rates(obs):
    due_age = np.where(obs["already_stopped"], DUE_AGE_EXTRA_STOP, DUE_AGE_FIRST_STOP)
    obs = obs.assign(tyres_due=obs["tyre_age"] >= due_age)
    return obs.groupby(["already_stopped", "tyres_due"])["pitted"].agg(p_pit="mean", n="size").reset_index()


if __name__ == "__main__":
    rows = []
    for year, rnd in RACES:
        s = fastf1.get_session(year, rnd, "R")
        s.load(telemetry=False, weather=False, messages=False)
        periods = safety_car_periods(s.laps)
        for r in reactions(s.laps, periods):
            rows.append(dict(race=f"{year} {s.event['EventName']}", **r))
        print(f"{year} {s.event['EventName']}: safety cars from lap {[p[0] for p in periods]}")

    obs = pd.DataFrame(rows)
    rates = pit_rates(obs)
    print(f"\n{obs.groupby(['race', 'lap']).ngroups} safety cars, {len(obs)} car observations")
    print(rates.round(2).to_string(index=False))
    rates.to_csv("data/sc_pit_rates.csv", index=False)

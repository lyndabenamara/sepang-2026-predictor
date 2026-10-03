# sepang_strategies.py: the race plans cars could choose from at Sepang
# Baku had five years of real strategies to copy. Sepang has none with these cars,
# so every legal plan gets built and timed from the measured tyre numbers instead.
# Produces: data/sepang_strategies.csv, the best version of every plan and its cost

import itertools
import numpy as np
import pandas as pd

LAPS = 56
PIT_LOSS = 22.5                  # Pirelli's estimate for Sepang
MIN_STINT = 5                    # nobody pits for fresh tyres after fewer laps than this
SETS = {"SOFT": 3, "MEDIUM": 3, "HARD": 2}   # sets usually left for the race once qualifying has used some softs

# How choosy teams are. A plan that's this many seconds slower than the best one is
# picked about a third as often. ASSUMPTION: 5s, roughly how unsure we are of the
# wear numbers over a whole race. It's what stops every car doing the identical plan.
CHOOSINESS = 5.0


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: EVERY LEGAL PLAN
# ════════════════════════════════════════════════════════════════════════════
# 1, 2 or 3 stops, at least two different compounds (it's the rule in a dry race),
# and never more sets of a compound than a car actually has.

def legal_sequences(max_stops=3):
    seqs = []
    for stops in range(1, max_stops + 1):
        for seq in itertools.product(SETS, repeat=stops + 1):
            if len(set(seq)) < 2:
                continue
            if any(seq.count(c) > n for c, n in SETS.items()):
                continue
            seqs.append(seq)
    return seqs


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: THE BEST STOP LAPS FOR EACH PLAN
# ════════════════════════════════════════════════════════════════════════════
# Time lost over a stint of L laps = compound offset x L + wear x (1 + 2 + ... + L)
# Try every way of splitting the race and keep the cheapest. This only counts tyres
# and pit stops; traffic and track position are left to the race simulator.

def stint_cost(comp, length, wear, offset):
    return offset[comp] * length + wear[comp] * length * (length + 1) / 2


def best_split(seq, wear, offset, laps=LAPS):
    n = len(seq)
    best = (np.inf, None)
    for stops in itertools.combinations(range(MIN_STINT, laps - MIN_STINT + 1), n - 1):
        bounds = (0,) + stops + (laps,)
        lengths = np.diff(bounds)
        if lengths.min() < MIN_STINT:
            continue
        cost = sum(stint_cost(c, L, wear, offset) for c, L in zip(seq, lengths)) + PIT_LOSS * (n - 1)
        if cost < best[0]:
            best = (cost, stops)
    return best


def plan_table(wear, offset, laps=LAPS):
    rows = []
    for seq in legal_sequences():
        cost, stops = best_split(seq, wear, offset, laps)
        rows.append(dict(compounds=seq, stop_laps=stops, n_stops=len(seq) - 1,
                         race_laps=laps, cost=cost))
    t = pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)
    t["vs_best"] = t["cost"] - t["cost"].min()
    return t


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: HOW LIKELY EACH CAR IS TO PICK EACH PLAN
# ════════════════════════════════════════════════════════════════════════════
# The plans are fixed, but a car that's gentle on its tyres finds the longer stints
# cheaper, so it leans towards fewer stops. Each car's chances come from timing every
# plan with ITS wear rate, then favouring the quick ones.

def plan_probabilities(plans, wear, offset, deg_mult):
    probs = []
    for m in deg_mult:
        w = {c: v * m for c, v in wear.items()}
        cost = np.array([
            sum(stint_cost(c, L, w, offset) for c, L in zip(row.compounds, np.diff((0,) + tuple(row.stop_laps) + (LAPS,))))
            + PIT_LOSS * row.n_stops
            for row in plans.itertuples()])
        p = np.exp(-(cost - cost.min()) / CHOOSINESS)
        probs.append(p / p.sum())
    return np.array(probs)


if __name__ == "__main__":
    tyres = pd.read_csv("data/sepang_tyres.csv", index_col="compound")
    wear, offset = tyres["wear"].to_dict(), tyres["offset"].to_dict()
    plans = plan_table(wear, offset)
    plans.to_pickle("data/sepang_strategies.pkl")
    out = plans.assign(compounds=plans.compounds.map("-".join), stop_laps=plans.stop_laps.map(str))
    out.to_csv("data/sepang_strategies.csv", index=False)
    print(f"{len(plans)} legal plans. Fastest 12 (seconds lost to tyres + stops):")
    print(out.head(12)[["compounds", "stop_laps", "n_stops", "cost", "vs_best"]].round(1).to_string(index=False))
    print("\nBest plan for each number of stops:")
    print(out.groupby("n_stops").first()[["compounds", "stop_laps", "vs_best"]].round(1).to_string())

    # What an average car, a tyre-friendly car and a tyre-hungry car would pick
    p = plan_probabilities(plans, wear, offset, [1.0, 0.4, 1.4])
    for label, row in zip(["average car", "gentle (McLaren-like 0.4x)", "hungry (1.4x)"], p):
        mix = pd.Series(row).groupby(plans["n_stops"]).sum()
        print(f"  {label:<28} " + "  ".join(f"{k}-stop {v:.0%}" for k, v in mix.items()))

# sepang_telemetry.py: two comparison charts of the four top teams at Sepang.
#   1. Speed trace of each team's best qualifier on their fastest Q3 lap (the one
#      that set their grid slot), plus where they gain or lose time to pole.
#   2. Tyre wear from Friday's long runs, one panel per team.
# These are for the video, the prediction doesn't use them.
# Produces: data/sepang_q3_speed_trace.png and data/sepang_longrun_wear.png

import warnings
import numpy as np
import matplotlib.pyplot as plt
import fastf1
from fastf1 import utils
from scipy.signal import find_peaks

import sepang_tyres

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("cache")
fastf1.set_log_level("ERROR")

ROUND = 16
DRIVERS = {"VER": "Red Bull Racing", "HAM": "Ferrari", "ANT": "Mercedes", "NOR": "McLaren"}
# Team colours, checked for colourblind separation. Mercedes is a darker teal than
# their usual one so it shows up on a white background.
COLOUR = {"Red Bull Racing": "#3671C6", "Ferrari": "#E8002D", "Mercedes": "#00A19C", "McLaren": "#FF8000"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


# ════════════════════════════════════════════════════════════════════════════
# CORNERS
# ════════════════════════════════════════════════════════════════════════════
# FastF1 has no corner map for Sepang in any year, so corners are found from the
# telemetry itself: every big dip in speed is a braking zone. Each dip was checked
# against the distance windows in the FIA's 2026 Power Unit document for this race
# (T1-T2 600-750m, T9-T11 3100-3400m, T15 exit 5110m), so the numbers aren't guessed.
# T3, T5, T10, T12 and T13 are taken flat or nearly flat, so they don't show as dips.

# The two fast corner sections, from the same FIA document. These are where the
# tyres carry the most load for the longest, which is what overheats them here.
FAST_SECTIONS = [(1900, 2350, "T5-T6"), (3750, 4000, "T12-T13")]


def braking_zones(tel):
    grid = np.arange(0, int(tel["Distance"].max()), 5)
    speed = np.interp(grid, tel["Distance"], tel["Speed"])
    dips, _ = find_peaks(-speed, prominence=12, distance=20)
    # T6 is a shallow dip (around 245 km/h) that some laps don't show, so only
    # the slow corners are matched by count and T6 is placed from the FIA window.
    slow = [grid[i] for i in dips if speed[i] < 220]
    if len(slow) != 7:
        raise RuntimeError(f"Expected 7 slow corners, found {len(slow)}: check the reference lap")
    return dict(zip(["T1", "T4", "T7-8", "T9", "T11", "T14", "T15"], slow)) | {"T6": 2130}


# ════════════════════════════════════════════════════════════════════════════
# CHART 1: Q3 SPEED TRACE
# ════════════════════════════════════════════════════════════════════════════

def q3_laps():
    q = fastf1.get_session(2026, ROUND, "Q")
    q.load(telemetry=True, weather=False, messages=True)
    q3 = q.laps.split_qualifying_sessions()[2]
    return {d: q3.pick_drivers(d).pick_fastest() for d in DRIVERS}


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def speed_trace(laps):
    ref = laps["VER"]
    tel = {d: lap.get_car_data().add_distance() for d, lap in laps.items()}
    corners = braking_zones(tel["VER"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        style(ax)
        for lo, hi, _ in FAST_SECTIONS:
            ax.axvspan(lo, hi, color="#f2efe6", zorder=0)
        for dist in corners.values():
            ax.axvline(dist, color=MUTED, linestyle=":", linewidth=0.8, zorder=1)

    for d, t in tel.items():
        team = DRIVERS[d]
        ax1.plot(t["Distance"], t["Speed"], color=COLOUR[team], linewidth=1.6,
                 label=f"{d}  {team}  {fmt(laps[d]['LapTime'])}")
    ax1.set_ylabel("Speed (km/h)", color=INK)
    ax1.set_ylim(40, 345)
    for name, dist in corners.items():
        ax1.text(dist, 46, name, fontsize=8, color=MUTED, ha="center")
    for lo, hi, name in FAST_SECTIONS:
        ax1.text((lo + hi) / 2, 335, f"{name}\nfast corners", fontsize=8, color=MUTED,
                 ha="center", va="top")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    ax1.set_title("Sepang 2026 qualifying: fastest Q3 lap from each top team\n"
                  "Shaded = the fast corner sections, where the tyres are loaded hardest",
                  loc="left", fontsize=12, color=INK)

    # Running gap to pole: above zero = behind VER at that point of the lap
    ax2.axhline(0, color=INK, linewidth=0.8)
    for d in DRIVERS:
        if d == "VER":
            continue
        delta, ref_tel, _ = utils.delta_time(ref, laps[d])
        ax2.plot(ref_tel["Distance"], delta, color=COLOUR[DRIVERS[d]], linewidth=1.6)
        ax2.text(ref_tel["Distance"].iloc[-1] + 30, delta.iloc[-1], d, fontsize=9,
                 color=INK, va="center")
    ax2.set_ylabel("Gap to VER (s)\nabove 0 = behind", color=INK)
    ax2.set_xlabel("Distance around the lap (m)", color=INK)
    ax2.set_xlim(0, ref.get_car_data().add_distance()["Distance"].max() + 150)

    fig.tight_layout()
    fig.savefig("data/sepang_q3_speed_trace.png", dpi=150)
    plt.close(fig)
    return tel, corners


def fmt(lap_time):
    s = lap_time.total_seconds()
    return f"{int(s // 60)}:{s % 60:06.3f}"


def where_time_goes(laps, corners):
    """Gap to VER at the end of each part of the lap, so the chart can be read out loud."""
    ref = laps["VER"]
    marks = [("T1-T4", corners["T4"]), ("T5-T8 (fast)", corners["T7-8"]),
             ("T9-T11", corners["T11"]), ("T12-T14 (fast)", corners["T14"]), ("T15 to the line", None)]
    print("\nGap to VER at each point of the lap (seconds, + = behind):")
    print(f"  {'':>4}" + "".join(f"{m:>17}" for m, _ in marks))
    for d in DRIVERS:
        if d == "VER":
            continue
        delta, ref_tel, _ = utils.delta_time(ref, laps[d])
        dist = ref_tel["Distance"].values
        row = [delta.values[-1] if at is None else np.interp(at, dist, delta.values) for _, at in marks]
        print(f"  {d:>4}" + "".join(f"{v:>+17.3f}" for v in row))


# ════════════════════════════════════════════════════════════════════════════
# CHART 2: FRIDAY LONG RUNS
# ════════════════════════════════════════════════════════════════════════════
# Same long runs and same wear fit as sepang_tyres.py. Each run is shown relative to
# its own starting pace, so fuel loads and engine modes between runs don't matter,
# only how fast the lap times drift up as the tyres wear.

def longrun_wear():
    runs = sepang_tyres.long_runs()
    wear = sepang_tyres.fit_wear(runs, group="Team")
    runs = runs[runs["Team"].isin(COLOUR)]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, team in zip(axes, COLOUR):
        style(ax)
        r = runs[runs["Team"] == team]
        rate = wear.loc[team, "wear"]
        for _, g in r.groupby("run"):
            age = g["TyreLife"].values
            # Baseline = where this run would sit at tyre age 0 under the team's wear rate
            base = np.mean(g["t"].values - rate * age)
            ax.plot(age, g["t"].values - base, "o", color=COLOUR[team], markersize=4.5,
                    alpha=0.55, markeredgecolor="white", markeredgewidth=0.6)
        x = np.array([r["TyreLife"].min(), r["TyreLife"].max()])
        ax.plot(x, rate * x, color=COLOUR[team], linewidth=2)
        ax.set_title(f"{team}\n{rate:.2f} s per lap  ({r['run'].nunique()} runs)",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 20)
        ax.set_xlabel("Tyre age (laps)", color=INK)
    axes[0].set_ylabel("Time lost to tyre wear (s)", color=INK)
    fig.suptitle("Sepang 2026 Friday long runs: how fast each top team's lap times rose as the tyres wore\n"
                 "Practice fuel loads and run plans are unknown. In 2026 so far, Friday tyre wear "
                 "hasn't predicted Sunday's (correlation 0.04)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig("data/sepang_longrun_wear.png", dpi=150)
    plt.close(fig)
    return wear.loc[list(COLOUR)]


if __name__ == "__main__":
    laps = q3_laps()
    tel, corners = speed_trace(laps)
    print("Fastest Q3 laps:")
    for d, lap in laps.items():
        print(f"  {d} {DRIVERS[d]:<16} {fmt(lap['LapTime'])}  on {lap['Compound']}, "
              f"top speed {tel[d]['Speed'].max():.0f} km/h")
    where_time_goes(laps, corners)
    print("\nSaved: data/sepang_q3_speed_trace.png")

    wear = longrun_wear()
    print("\nFriday tyre wear, seconds lost per lap of tyre age (fuel burn hides ~0.05 for everyone):")
    print(wear.round(3).to_string())
    print("\nSaved: data/sepang_longrun_wear.png")

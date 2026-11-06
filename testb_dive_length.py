"""
Test B - How much OWTT range error does CSAC drift cause over a dive?

Setup (matches the lab):
  - Deck clock: stays locked to GPS for the whole mission, so it does not drift.
    Its only error is a few ns of GPS timing noise (DECK_NOISE_S, assumed).
  - Vehicle clock: a CSAC SA65, lined up with the deck clock before launch,
    then free-running for the whole dive (no GPS underwater).
OWTT range error = speed_of_sound x (vehicle clock error - deck clock error).

We run many random dives (Monte Carlo) and report the RMS range error at
several dive lengths, split by source, for two ways of preparing the vehicle
clock on deck before launch:

  1. "Synced to deck clock"     - 1PPS sync command only: time lined up to
                                  +/-100 ns, speed left at factory calibration
                                  (+/-5e-11). Datasheet values.
  2. "Disciplined to deck clock" - left connected long enough to also correct
                                  its speed (ASSUMED +/-5 ns, +/-1e-12).
In both cases the vehicle clock runs on its own once it launches.

No HoloOcean needed: geometry does not change the clock error.
Temperature is NOT included here (that is a later test).

Outputs (in ./results): test_b_dive_length.png, test_b_dive_length.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from clock_model import ClockParams, csac_sa65, simulate_phase, SECONDS_PER_MONTH

OUT = "results"
SOUND_SPEED = 1500.0           # m/s
N_RUNS = 500                   # random dives per case
DT_S = 10.0                    # sample spacing (exact model, so 10 s is fine)
MAX_DIVE_H = 24
REPORT_H = [1, 6, 12, 24]
SEED = 7

# Starting conditions (uniform random within +/- these limits, independently per clock)
SCENARIOS = {
    "Synced to deck clock": dict(x0=100e-9, y0=5e-11),      # datasheet: 1PPS sync, shipment accuracy
    "Disciplined to deck clock": dict(x0=5e-9, y0=1e-12),   # ASSUMED - measure this on the bench!
}
AGING_LIMIT = 9e-10 / SECONDS_PER_MONTH   # datasheet: < 9e-10 per month, sign unknown
DECK_NOISE_S = 5e-9                       # ASSUMED GPS-locked deck clock timing noise (1 sigma)

COLORS = {"Sync offset": "#2a78d6", "Frequency offset": "#eb6834",
          "Aging": "#1baf7a", "Random noise (q1, q2)": "#eda100",
          "Deck clock (GPS noise)": "#e87ba4", "Total": "#1f1f1e"}


def relative_error(rng, x0_lim, y0_lim, aging_lim, noise, deck_noise):
    """Vehicle minus deck clock error, shape (N_RUNS, n). Only the chosen parts switched on."""
    p = csac_sa65()                                   # vehicle clock
    p.x0 = rng.uniform(-x0_lim, x0_lim, (N_RUNS, 1))
    p.y0 = rng.uniform(-y0_lim, y0_lim, (N_RUNS, 1))
    p.aging = rng.uniform(-aging_lim, aging_lim, (N_RUNS, 1))
    t, x_vehicle = simulate_phase(p, MAX_DIVE_H * 3600, DT_S, n_runs=N_RUNS, rng=rng,
                                  include_noise=noise)
    # Deck clock is locked to GPS: no drift, just a few ns of noise each ping
    x_deck = rng.normal(0.0, deck_noise, x_vehicle.shape) if deck_noise > 0 else 0.0
    return t, x_vehicle - x_deck


def rms_range_mm(dx):
    return np.sqrt(np.mean(dx**2, axis=0)) * SOUND_SPEED * 1000.0


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}
    rows = []

    for scen, lim in SCENARIOS.items():
        parts = {
            "Sync offset":            (lim["x0"], 0, 0, False, 0),
            "Frequency offset":       (0, lim["y0"], 0, False, 0),
            "Aging":                  (0, 0, AGING_LIMIT, False, 0),
            "Random noise (q1, q2)":  (0, 0, 0, True, 0),
            "Deck clock (GPS noise)": (0, 0, 0, False, DECK_NOISE_S),
            "Total":                  (lim["x0"], lim["y0"], AGING_LIMIT, True, DECK_NOISE_S),
        }
        results[scen] = {}
        for name, (x0, y0, ag, noise, deck) in parts.items():
            t, dx = relative_error(rng, x0, y0, ag, noise, deck)
            results[scen][name] = rms_range_mm(dx)
        hours = t / 3600

        print(f"\n=== {scen} ===   RMS range error (mm), vehicle CSAC vs GPS-locked deck clock")
        print(f"{'source':<24}" + "".join(f"{h:>8} h" for h in REPORT_H))
        for name, curve in results[scen].items():
            vals = [curve[np.argmin(abs(hours - h))] for h in REPORT_H]
            print(f"{name:<24}" + "".join(f"{v:10.3f}" for v in vals))
            rows.append([scen, name] + vals)

    with open(os.path.join(OUT, "test_b_dive_length.csv"), "w") as f:
        f.write("scenario,source," + ",".join(f"rms_mm_{h}h" for h in REPORT_H) + "\n")
        for r in rows:
            f.write(",".join([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]]) + "\n")

    # ---- plot: one panel per scenario, same y-scale so they compare directly ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, scen in zip(axes, SCENARIOS):
        for name, curve in results[scen].items():
            lw = 2.2 if name == "Total" else 2
            ls = "--" if name == "Total" else "-"     # dashed so lines under it stay visible
            ax.plot(hours[1:], curve[1:], ls, lw=lw, color=COLORS[name], label=name)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 20)
        ax.set_xlim(0, MAX_DIVE_H)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Dive time since sync (h)")
        ax.set_title(scen, loc="left")
        ax.grid(True, which="both", color="#e4e3dc", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("RMS OWTT range error (mm, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=6)
    fig.suptitle("Range error from vehicle CSAC drift vs GPS-locked deck clock "
                 "(500 random dives, no temperature)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "test_b_dive_length.png"), dpi=130)
    print(f"\nSaved {OUT}/test_b_dive_length.png and {OUT}/test_b_dive_length.csv")


if __name__ == "__main__":
    main()
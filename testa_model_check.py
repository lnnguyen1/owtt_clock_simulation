"""
Test A - Does the simulated CSAC match its datasheet Allan deviation?

1. Simulate one CSAC SA65 for 3 days, sampled every 1 s.
2. Compute its overlapping Allan deviation.
3. Compare with (a) the datasheet values and (b) the model's own prediction.
4. Fit q1, q2 back from the simulated ADEV - the same fit you will later run
   on real bench data - and check it recovers the values we put in.

Outputs (in ./results):  test_a_adev.png, test_a_adev.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from clock_model import csac_sa65, simulate_phase, CSAC_SA65_SPEC_ADEV
from allan import overlapping_adev, fit_q1_q2

OUT = "results"
DURATION_S = 3 * 86400
DT_S = 1.0
SEED = 1

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6f6e69"


def main():
    os.makedirs(OUT, exist_ok=True)
    clock = csac_sa65()
    t, x = simulate_phase(clock, DURATION_S, DT_S, rng=np.random.default_rng(SEED))
    x = x[0]

    taus, adev, nterms = overlapping_adev(x, DT_S)
    q1_fit, q2_fit = fit_q1_q2(taus, adev)

    # ---- table at the datasheet averaging times ----
    spec_t = np.array(sorted(CSAC_SA65_SPEC_ADEV))
    _, adev_at_spec, _ = overlapping_adev(x, DT_S, taus=spec_t)
    print("\nAllan deviation at datasheet averaging times")
    print(f"{'tau (s)':>8} {'datasheet':>11} {'model':>11} {'simulated':>11}  result")
    for tau, sim in zip(spec_t, adev_at_spec):
        spec = CSAC_SA65_SPEC_ADEV[tau]
        model = clock.adev_theory(tau)
        ok = "PASS" if sim <= 1.15 * spec else "CHECK"   # 15 % allowance for random scatter
        print(f"{tau:8.0f} {spec:11.2e} {model:11.2e} {sim:11.2e}  {ok}")

    print("\nFit of q1, q2 from the simulated ADEV (paper eq. 47)")
    print(f"  q1: put in {clock.q1:.2e} s    recovered {q1_fit:.2e} s")
    print(f"  q2: put in {clock.q2:.2e} /s   recovered {q2_fit:.2e} /s")
    print("  (q2 is hard to recover: its effect only shows at long tau, where few samples exist)")

    np.savetxt(os.path.join(OUT, "test_a_adev.csv"),
               np.column_stack([taus, adev, clock.adev_theory(taus), nterms]),
               delimiter=",", header="tau_s,adev_simulated,adev_model,n_terms", comments="")

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    tt = np.logspace(0, np.log10(taus[-1]), 200)
    ax1.loglog(tt, clock.adev_theory(tt), "--", color=MUTED, lw=1.5,
               label="Model prediction  √(q1/τ + q2·τ/3)")
    ax1.loglog(taus, adev, "-o", color=BLUE, lw=2, ms=5, label="Simulated CSAC (3 days)")
    ax1.loglog(spec_t, [CSAC_SA65_SPEC_ADEV[s] for s in spec_t], "s", color=ORANGE,
               ms=9, label="Datasheet limit")
    ax1.set_xlabel("Averaging time τ (s)")
    ax1.set_ylabel("Allan deviation σy(τ)")
    ax1.set_title("CSAC SA65 stability: simulation vs datasheet", loc="left")
    ax1.grid(True, which="both", color="#e4e3dc", lw=0.6)
    ax1.legend(frameon=False, fontsize=9)

    hours = t / 3600
    ax2.plot(hours, x * 1e6, color=BLUE, lw=2)
    ax2.set_xlabel("Time since sync (h)")
    ax2.set_ylabel("Phase error x(t) (µs)")
    ax2.set_title("Same clock: time error over 3 days", loc="left")
    ax2.grid(True, color="#e4e3dc", lw=0.6)
    ax2.text(0.02, 0.04, "Slow wander = random-walk frequency noise (q2) + aging\n"
             "Fine wiggle = white frequency noise (q1)",
             transform=ax2.transAxes, va="bottom", fontsize=9, color=MUTED)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "test_a_adev.png"), dpi=130)
    print(f"\nSaved {OUT}/test_a_adev.png and {OUT}/test_a_adev.csv")


if __name__ == "__main__":
    main()
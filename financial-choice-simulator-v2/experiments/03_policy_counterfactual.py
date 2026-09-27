"""
Experiment 2: does combining a cash transfer with reduced administrative
friction ("sludge eradication") beat either intervention alone?

Design note: the SAME households, under the SAME simulated shock histories,
are run under four policy conditions (paired / common random numbers). This
is a deliberately different, more defensible design than comparing four
separate random samples: it removes population-sampling noise from the
comparison, and it lets us bootstrap the interaction effect directly from
the paired outcomes without re-running the (slower) simulation each time.
"""
import json
import os
import sys
import numpy as np

# Make sure the project root (the folder that contains "src/") is on the
# import path, no matter what directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (Params, Household, ChoiceSet, default_actions,
                        sample_population, generate_offer_path,
                        NoIntervention, LiquidityTransfer, sludge_reduced_params,
                        PAYDAY, COLLAPSE)
from src.model.shocks import generate_shock_path
from src.estimation.stats import proportion_ci, interaction_effect

DEFAULT_SHOCK_P_START = 0.02  # used only if no estimation results are found at all

N = 4000
HORIZON = 12  # matches the 12-month recall window used throughout this project
SHOCK_SEVERITY = 0.35
CASH_AMOUNT = 500.0
CASH_MONTH = 3
REDUCED_COG_REFI = 0.20  # "sludge eradication": cut the cognitive cost of refinancing

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "estimated_params_real.json")
SYNTHETIC_RESULTS_PATH = os.path.join(RESULTS_DIR, "estimated_params.json")


def load_estimated_params():
    """Returns (Params, shock_p_start)."""
    if os.path.exists(REAL_RESULTS_PATH):
        with open(REAL_RESULTS_PATH) as f:
            d = json.load(f)
        print(f"Using REAL-data-calibrated parameters from {REAL_RESULTS_PATH}:")
        print(f"  alpha={d['alpha']:.4f}  cog_payday={d['cog_payday']:.4f}  "
              f"shock_p_start={d['shock_p_start']:.4f}")
        print(f"  (calibrated against {d.get('calibration_source', 'unknown source')})")
        return Params(alpha=d["alpha"], cog_payday=d["cog_payday"]), d["shock_p_start"]
    if os.path.exists(SYNTHETIC_RESULTS_PATH):
        with open(SYNTHETIC_RESULTS_PATH) as f:
            d = json.load(f)
        print(f"No real-data results found. Using SYNTHETIC-validation parameters "
              f"from {SYNTHETIC_RESULTS_PATH}:")
        print(f"  alpha={d['alpha']:.4f}  cog_payday={d['cog_payday']:.4f}  "
              f"shock_p_start={d['shock_p_start']:.4f}")
        print("  (these came from the ground-truth recovery check, not real data --")
        print("   run experiments/02_estimate_from_real_shed.py for a real-data result)")
        return Params(alpha=d["alpha"], cog_payday=d["cog_payday"]), d["shock_p_start"]
    print("No results/estimated_params_real.json or estimated_params.json found.")
    print("Run experiments/02_estimate_from_real_shed.py first for a real-data result")
    print("-- falling back to the defaults for now.\n")
    return Params(), DEFAULT_SHOCK_P_START


def run_cohort(params, intervention, pop, shocks, offers) -> np.ndarray:
    cs = ChoiceSet(default_actions(params), params)
    captured = np.zeros(N, dtype=bool)
    for i in range(N):
        hh = Household(pop["income"][i], pop["obligations"][i], pop["balance"][i],
                        pop["debt"][i], pop["alpha"][i], params)
        for t in range(HORIZON):
            income_t = hh.income * (1 - SHOCK_SEVERITY) if shocks[i][t] else hh.income
            transfer = intervention.transfer(t, hh)
            ctx = hh.begin_period(t, income_t, transfer, bool(offers[i][t]))
            dec = cs.choose(hh, ctx)
            hh.apply_action(dec.chosen, ctx)
            if dec.chosen in (PAYDAY, COLLAPSE):
                captured[i] = True
    return captured


def main():
    base_params, shock_p_start = load_estimated_params()
    sludge_params = sludge_reduced_params(base_params, REDUCED_COG_REFI)

    rng = np.random.default_rng(42)
    pop = sample_population(N, rng)
    shocks = [generate_shock_path(HORIZON, np.random.default_rng(rng.integers(1_000_000_000)),
                                   p_start=shock_p_start) for _ in range(N)]
    offers = [generate_offer_path(HORIZON, np.random.default_rng(rng.integers(1_000_000_000)))
              for _ in range(N)]

    print(f"Simulating {N} households x {HORIZON} months x 4 policy arms...")
    print("Running Control...")
    c_control = run_cohort(base_params, NoIntervention(), pop, shocks, offers)
    print("Running Cash Transfer only...")
    c_cash = run_cohort(base_params, LiquidityTransfer(CASH_AMOUNT, CASH_MONTH), pop, shocks, offers)
    print("Running Sludge Reduction only...")
    c_sludge = run_cohort(sludge_params, NoIntervention(), pop, shocks, offers)
    print("Running Both...")
    c_both = run_cohort(sludge_params, LiquidityTransfer(CASH_AMOUNT, CASH_MONTH), pop, shocks, offers)

    def summarize(name, arr):
        p = arr.mean()
        lo, hi = proportion_ci(p, N)
        print(f"  {name:<20s} capture rate = {p:>6.1%}   95% CI [{lo:.1%}, {hi:.1%}]   n={N}")
        return p

    print("\nResults:")
    p_control = summarize("Control", c_control)
    p_cash = summarize("Cash transfer", c_cash)
    p_sludge = summarize("Sludge reduction", c_sludge)
    p_both = summarize("Both", c_both)

    print(f"\nEffect of cash alone:    {p_cash - p_control:+.1%}")
    print(f"Effect of sludge alone:  {p_sludge - p_control:+.1%}")
    print(f"Effect of both together: {p_both - p_control:+.1%}")

    inter, se, pval = interaction_effect(p_control, N, p_cash, N, p_sludge, N, p_both, N)
    print(f"\nInteraction effect (treating the 4 arms as independent samples): "
          f"{inter:+.4f}  SE={se:.4f}  p={pval:.4f}")
    print("(This is a CONSERVATIVE approximation here, since the arms actually share")
    print(" households/shocks -- see the paired bootstrap below for the real test.)")

    print("\nPaired bootstrap on the interaction effect (2,000 resamples, instant --")
    print("this just resamples the outcomes already computed above, no resimulation)...")
    boot_rng = np.random.default_rng(7)
    draws = np.empty(2000)
    for b in range(2000):
        idx = boot_rng.integers(0, N, size=N)
        pc = c_control[idx].mean(); pca = c_cash[idx].mean()
        ps = c_sludge[idx].mean(); pb = c_both[idx].mean()
        draws[b] = pb - pca - ps + pc
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"95% CI on the interaction effect: [{lo:+.4f}, {hi:+.4f}]")
    if hi < 0:
        print("-> Interval is entirely below zero: in THIS simulation, combining the")
        print("   two interventions beats the sum of their separate effects.")
    elif lo > 0:
        print("-> Interval is entirely above zero (anti-complementary): combining them")
        print("   does WORSE than the sum of their separate effects.")
    else:
        print("-> Interval includes zero: this simulation does not give clear evidence")
        print("   either way about complementarity.")


if __name__ == "__main__":
    main()

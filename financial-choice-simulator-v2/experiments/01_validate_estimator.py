"""
Experiment 1: does the SMM estimator actually recover the right answer?

We validate the THREE-parameter estimator (alpha, cog_payday, shock_p_start)
against a known, hidden ground truth before ever trusting what it recovers
from real data in experiment 2. Pick a known theta, generate synthetic data
from it, hide the theta, and see if the estimator finds its way back.

Run this first. It writes results/estimated_params.json (the SYNTHETIC-
validation result -- experiments/03_policy_counterfactual.py prefers the
real-data result from experiment 2 if that also exists).
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Params
from src.estimation import build_sim_inputs, simulate_moments, run_smm, bootstrap_se

TRUE_ALPHA = 0.55
TRUE_COG_PAYDAY = 0.20
TRUE_SHOCK_P_START = 0.02
N_TRUTH = 400
N_ESTIMATION = 250
BOOTSTRAP_B = 20

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "estimated_params.json")


def main():
    base_params = Params()
    true_theta = (TRUE_ALPHA, TRUE_COG_PAYDAY, TRUE_SHOCK_P_START)

    print("=" * 70)
    print("STEP 1: generate synthetic 'ground truth' data")
    print("=" * 70)
    print(f"True (hidden) parameters: alpha={TRUE_ALPHA}, cog_payday={TRUE_COG_PAYDAY}, "
          f"shock_p_start={TRUE_SHOCK_P_START}")
    truth_sim = build_sim_inputs(n=N_TRUTH, seed=999)
    target = simulate_moments(true_theta, truth_sim, base_params)
    print(f"Target moments computed from that ground truth: {target}")
    print("  m1 = share of households ever forced into payday debt / collapse")
    print("  m2 = share of shortfall-months resolved by deliberately choosing payday debt")
    print("  m3 = share of all household-months resolved by deferring a payment")

    print("\n" + "=" * 70)
    print("STEP 2: run the estimator WITHOUT telling it the true parameters")
    print("=" * 70)
    est_sim = build_sim_inputs(n=N_ESTIMATION, seed=1)
    result = run_smm(est_sim, base_params, target, seed=0)
    alpha_hat, cog_payday_hat, p_start_hat = result.x
    fitted = simulate_moments(result.x, est_sim, base_params)
    print(f"Recovered:  alpha={alpha_hat:.4f}   cog_payday={cog_payday_hat:.4f}   "
          f"shock_p_start={p_start_hat:.4f}")
    print(f"True:       alpha={TRUE_ALPHA:.4f}   cog_payday={TRUE_COG_PAYDAY:.4f}   "
          f"shock_p_start={TRUE_SHOCK_P_START:.4f}")
    print(f"Final SSE: {result.fun:.6f}")
    print("\nFit table (target vs. fitted moments):")
    print(f"  {'moment':<8}{'target':>10}{'fitted':>10}{'abs error':>12}")
    for i, name in enumerate(["m1", "m2", "m3"]):
        print(f"  {name:<8}{target[i]:>10.4f}{fitted[i]:>10.4f}{abs(target[i]-fitted[i]):>12.4f}")

    print("\n" + "=" * 70)
    print(f"STEP 3: bootstrap standard errors (B={BOOTSTRAP_B} resamples)")
    print("=" * 70)
    mean_theta, se_theta, draws = bootstrap_se(est_sim, base_params, target, B=BOOTSTRAP_B)
    print(f"\nBootstrap mean:  alpha={mean_theta[0]:.4f} (SE {se_theta[0]:.4f})   "
          f"cog_payday={mean_theta[1]:.4f} (SE {se_theta[1]:.4f})   "
          f"shock_p_start={mean_theta[2]:.4f} (SE {se_theta[2]:.4f})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "alpha": float(alpha_hat),
        "cog_payday": float(cog_payday_hat),
        "shock_p_start": float(p_start_hat),
        "alpha_bootstrap_se": float(se_theta[0]),
        "cog_payday_bootstrap_se": float(se_theta[1]),
        "shock_p_start_bootstrap_se": float(se_theta[2]),
        "note": "Recovered from SYNTHETIC ground-truth data, not real survey data. "
                "See README before treating these as real-world estimates.",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print("Next: run experiments/02_estimate_from_real_shed.py")


if __name__ == "__main__":
    main()

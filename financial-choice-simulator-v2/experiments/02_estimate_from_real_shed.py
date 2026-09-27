"""
Experiment 2: calibrate (alpha, cog_payday) against REAL Federal Reserve
SHED statistics, not synthetic ground truth.

Unlike experiment 1 (which validates the estimator against a KNOWN made-up
answer), this script's target moments are real, cited counts from the
published 2025 SHED codebook -- see src/data/shed_real_moments.py for the
exact variable names, counts, and source URL.

This is the script whose output should be reported as "the estimate" in any
write-up. Run experiments/01_validate_estimator.py first (or at least once)
so you have independent evidence the estimator itself is trustworthy before
trusting what it recovers here.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Params
from src.estimation import build_sim_inputs, simulate_moments, run_smm, bootstrap_se
from src.data.shed_real_moments import (
    real_target_vector, print_provenance, CODEBOOK_URL, CODEBOOK_SAMPLE_N,
)

N_ESTIMATION = 250
BOOTSTRAP_B = 20

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "estimated_params_real.json")


def main():
    print("=" * 70)
    print("Real Federal Reserve SHED calibration targets")
    print("=" * 70)
    print_provenance()
    target = real_target_vector()
    print(f"\nTarget vector used for calibration: {target}")

    base_params = Params()
    est_sim = build_sim_inputs(n=N_ESTIMATION, seed=1)

    print("\n" + "=" * 70)
    print("Running Differential Evolution against the real target moments...")
    print("=" * 70)
    result = run_smm(est_sim, base_params, target, seed=0)
    alpha_hat, cog_payday_hat, p_start_hat = result.x
    fitted = simulate_moments(result.x, est_sim, base_params)
    print(f"Recovered:  alpha={alpha_hat:.4f}   cog_payday={cog_payday_hat:.4f}   "
          f"shock_p_start={p_start_hat:.4f}")
    print(f"Final SSE: {result.fun:.6f}")
    print("\nFit table (real target vs. simulated, at the fitted parameters):")
    print(f"  {'moment':<24}{'target':>10}{'fitted':>10}{'abs error':>12}")
    for i, name in enumerate(["m1 (payday, 12mo)", "m2 (chronic revolver)", "m3 (deferred pmt)"]):
        print(f"  {name:<24}{target[i]:>10.4f}{fitted[i]:>10.4f}{abs(target[i]-fitted[i]):>12.4f}")

    print("\n" + "=" * 70)
    print(f"Bootstrap standard errors (B={BOOTSTRAP_B} resamples)...")
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
        "target_m1_payday_12mo": float(target[0]),
        "target_m2_chronic_revolver": float(target[1]),
        "target_m3_missed_min_payment": float(target[2]),
        "fitted_sse": float(result.fun),
        "calibration_source": CODEBOOK_URL,
        "calibration_sample_n": CODEBOOK_SAMPLE_N,
        "note": "Calibrated against REAL, cited 2025 SHED codebook tabulations "
                "(unweighted). See src/data/shed_real_moments.py for exact counts.",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print("Next: run experiments/03_policy_counterfactual.py")


if __name__ == "__main__":
    main()

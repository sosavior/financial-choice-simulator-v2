"""Runs the whole pipeline in order: tests, then estimation, then the
policy comparison. Just: python run_all.py"""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n" + "=" * 70)
    print(f"$ {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n'{' '.join(cmd)}' failed (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "experiments/01_validate_estimator.py"])
    run([sys.executable, "experiments/02_estimate_from_real_shed.py"])
    run([sys.executable, "experiments/03_policy_counterfactual.py"])
    print("\nAll done. See results/estimated_params_real.json for the fitted parameters.")

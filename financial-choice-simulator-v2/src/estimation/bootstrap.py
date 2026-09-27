"""Bootstrap standard errors for the SMM estimator.

Resamples the household population WITH replacement B times and re-runs
the full estimation on each resample. This is slow by nature (B full
optimizations) -- see README for how B, population size, and DE settings
trade off runtime against precision.
"""
from dataclasses import replace
from typing import Tuple
import numpy as np

from src.model import Params
from .smm import SimInputs, build_sim_inputs, simulate_moments, run_smm


def _resample(sim: SimInputs, rng: np.random.Generator) -> SimInputs:
    idx = rng.integers(0, sim.n, size=sim.n)
    pop = {k: v[idx] for k, v in sim.population.items()}
    shock_uniforms = [sim.shock_uniforms[i] for i in idx]
    offers = [sim.offers[i] for i in idx]
    return SimInputs(pop, shock_uniforms, offers, sim.n)


def bootstrap_se(base_sim: SimInputs, base_params: Params, target: np.ndarray,
                  B: int = 30, seed: int = 0, popsize: int = 8, maxiter: int = 20
                  ) -> Tuple[np.ndarray, np.ndarray, list]:
    """Returns (mean_theta, se_theta, all_theta_draws)."""
    rng = np.random.default_rng(seed)
    draws = []
    for b in range(B):
        resampled = _resample(base_sim, rng)
        res = run_smm(resampled, base_params, target, seed=1000 + b,
                       popsize=popsize, maxiter=maxiter)
        draws.append(res.x)
        print(f"  bootstrap {b + 1}/{B}: alpha={res.x[0]:.4f} cog_payday={res.x[1]:.4f} "
              f"shock_p_start={res.x[2]:.4f}")
    draws = np.array(draws)
    return draws.mean(axis=0), draws.std(axis=0, ddof=1), draws

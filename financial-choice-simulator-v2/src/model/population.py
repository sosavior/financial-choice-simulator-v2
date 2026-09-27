"""Generates a SYNTHETIC population of households for the simulation.

IMPORTANT: this is not real survey data. It draws income, debt, balance and
alpha from stated distributions (ASSUMED, see comments below) so that the
whole pipeline runs out of the box with zero external files.

If you later want to drive this from a real dataset (e.g. a household
finance survey extract you have permission to use), replace this function
with one that returns the same dict-of-arrays shape -- everything downstream
only depends on that shape, not on how the numbers were produced.
"""
from typing import Dict
import numpy as np

# Correlations between (income, obligation-ratio, debt, balance, alpha).
# ASSUMED, chosen so higher debt tends to come with higher obligations and
# lower savings -- a plausible household finance pattern, not a fitted one.
_CORR = np.array([
    [1.00, -0.20,  0.10,  0.40, -0.20],
    [-0.20, 1.00,  0.20, -0.20,  0.10],
    [0.10,  0.20,  1.00, -0.30,  0.10],
    [0.40, -0.20, -0.30,  1.00, -0.15],
    [-0.20, 0.10,  0.10, -0.15,  1.00],
])


def sample_population(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    z = rng.standard_normal((n, 5)) @ np.linalg.cholesky(_CORR).T

    income = np.exp(np.log(3200.0) + 0.35 * z[:, 0])
    debt = np.minimum(np.exp(np.log(6000.0) + 0.80 * z[:, 2]), 6.0 * income)

    max_ratio = 0.97 - 1.018 * 0.03 * debt / income
    obl_ratio = np.clip(np.exp(np.log(0.78) + 0.15 * z[:, 1]), 0.35, np.maximum(max_ratio, 0.35))

    balance = np.exp(np.log(1500.0) + 0.90 * z[:, 3])
    alpha = np.clip(0.50 + 0.15 * z[:, 4], 0.05, 1.20)

    return {
        "income": income,
        "obligations": income * obl_ratio,
        "debt": debt,
        "balance": balance,
        "alpha": alpha,
    }

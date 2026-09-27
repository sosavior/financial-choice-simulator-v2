"""Generates a month-by-month income-shock path for one household.

An income shock is a spell (one or more consecutive months) of reduced
income, e.g. a job loss or medical event. ASSUMED distribution -- see
README for how to replace with a real calibration source.
"""
import numpy as np


def shock_path_from_uniforms(u: np.ndarray, p_start: float, p_end: float = 0.20,
                              max_spell: int = 12) -> np.ndarray:
    """Turns a FIXED array of U(0,1) draws into a boolean shock path for a
    given (p_start, p_end). Because `u` doesn't depend on p_start, this lets
    p_start be treated as an estimated parameter under Common Random Numbers:
    the same underlying randomness is reused for every candidate p_start the
    optimizer tries -- see src/estimation/smm.py."""
    T = len(u)
    path = np.zeros(T, dtype=bool)
    in_shock, length = False, 0
    for t in range(T):
        if in_shock:
            in_shock = (u[t] >= p_end) and (length < max_spell)
        else:
            in_shock = u[t] < p_start
        length = length + 1 if in_shock else 0
        path[t] = in_shock
    return path


def generate_shock_path(T: int, rng: np.random.Generator, p_start: float = 0.05,
                         p_end: float = 0.20, max_spell: int = 12) -> np.ndarray:
    """Convenience one-shot version (draws its own randomness). Used where
    p_start is fixed rather than estimated -- e.g. tests."""
    return shock_path_from_uniforms(rng.random(T), p_start, p_end, max_spell)


def generate_offer_path(T: int, rng: np.random.Generator, p_offer: float = 0.20) -> np.ndarray:
    """Whether a refinancing offer is available to the household in month t."""
    return rng.random(T) < p_offer

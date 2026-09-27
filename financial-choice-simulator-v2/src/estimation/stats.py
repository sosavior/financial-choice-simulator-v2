"""Small, transparent statistics helpers. No hidden magic -- every formula
here is the standard textbook one, so you can check it by hand if you want."""
from typing import Tuple
import numpy as np
from scipy.stats import norm


def proportion_se(p: float, n: int) -> float:
    """Standard error of a sample proportion."""
    return float(np.sqrt(p * (1 - p) / n))


def proportion_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    se = proportion_se(p, n)
    return p - z * se, p + z * se


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
    """Tests H0: p1 == p2. Returns (z_statistic, two_sided_p_value)."""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p_value)


def interaction_effect(p_control: float, n_control: int,
                        p_cash: float, n_cash: int,
                        p_sludge: float, n_sludge: int,
                        p_both: float, n_both: int) -> Tuple[float, float, float]:
    """Difference-in-differences test for whether two interventions are
    complementary (interaction < 0, i.e. combined effect beats the sum of
    the parts) or merely additive (interaction ~ 0).

    interaction = (p_both - p_control) - (p_cash - p_control) - (p_sludge - p_control)
                = p_both - p_cash - p_sludge + p_control

    Returns (interaction_estimate, se, two_sided_p_value).
    """
    interaction = p_both - p_cash - p_sludge + p_control
    se = np.sqrt(
        p_control * (1 - p_control) / n_control
        + p_cash * (1 - p_cash) / n_cash
        + p_sludge * (1 - p_sludge) / n_sludge
        + p_both * (1 - p_both) / n_both
    )
    z = interaction / se if se > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(interaction), float(se), float(p_value)

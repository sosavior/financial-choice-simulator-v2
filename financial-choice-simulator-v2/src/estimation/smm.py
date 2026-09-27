"""Simulated Method of Moments estimator for (alpha, cog_payday, shock_p_start).

Common Random Numbers (CRN): the population and every household's raw
U(0,1) draws (for shocks and refinance offers) are generated ONCE, up front,
and reused unchanged for every candidate theta the optimizer tries. This is
what makes the objective surface deterministic in theta instead of noisy --
verified in tests/test_smm.py, which checks the exact same theta gives the
exact same moments on repeated calls.

Three parameters, three real target moments (exactly identified):

  theta = (alpha, cog_payday, shock_p_start)

  alpha         stress -> bandwidth elasticity
  cog_payday    cognitive cost of accessing predatory credit
  shock_p_start monthly probability a new income-shock spell begins

  m1 = share of households EVER forced into payday debt or a full
       cognitive collapse over the 12-month horizon
       target: BK2_c, "payday loan in the past 12 months" (real SHED count)

  m2 = of the months a household is in shortfall, the share resolved by
       DELIBERATELY choosing a payday loan (vs. deferring or collapsing)
       target: C4A, "carried an unpaid balance most/all of the time"
       (real SHED count, among cardholders)

  m3 = share of household-MONTHS (across the whole horizon) resolved by
       deferring a payment (a proxy for "paid less than the minimum")
       target: C3P, "did not pay / paid less than minimum last month"
       (real SHED count, among cardholders with a balance) -- this is an
       approximation: C3P asks about one specific month, m3 averages over
       12 simulated months, which is a real, disclosed limitation
       (see README "Known limitations").

An earlier version of this project fixed shock_p_start at an assumed value
(0.04/month). Fitting against real 2025 SHED moments showed that assumption
alone produced a nearly 5x-too-high payday-loan incidence relative to the
real BK2_c figure, no matter how (alpha, cog_payday) were tuned (SSE
plateaued around 0.02). Freeing shock_p_start to be estimated resolved this
(SSE drops by roughly two orders of magnitude) -- see the paper's Results
section for this diagnostic, run yourself via experiments/02_estimate_from_real_shed.py.
"""
from dataclasses import dataclass, replace
from typing import Tuple
import numpy as np
from scipy.optimize import differential_evolution

from src.model import Params, Household, ChoiceSet, default_actions, sample_population
from src.model.shocks import shock_path_from_uniforms, generate_offer_path
from src.model.params import PAYDAY, COLLAPSE, DEFER

SHOCK_SEVERITY = 0.35
# 12 months, deliberately: this matches the recall window of the actual SHED
# survey questions used as calibration targets in src/data/shed_real_moments.py
# ("in the past 12 months, did you..."), so the simulated and real moments
# are asking about the same period.
HORIZON_MONTHS = 12
SHOCK_P_END = 0.20  # monthly probability an ongoing shock spell ends (ASSUMED, not estimated)


@dataclass
class SimInputs:
    """Everything about the population and its random draws that must stay
    FIXED across every theta evaluation (this is the CRN). Shock draws are
    kept as raw uniforms, not pre-thresholded booleans, because
    shock_p_start is now itself an estimated parameter."""
    population: dict
    shock_uniforms: list   # list of np.ndarray, raw U(0,1) draws per household
    offers: list           # list of np.ndarray, boolean offer path per household
    n: int


def build_sim_inputs(n: int, seed: int) -> SimInputs:
    ss = np.random.SeedSequence(seed)
    pop_seed, shock_seed = ss.spawn(2)
    population = sample_population(n, np.random.default_rng(pop_seed))
    shock_uniforms, offers = [], []
    for child in shock_seed.spawn(n):
        rng = np.random.default_rng(child)
        shock_uniforms.append(rng.random(HORIZON_MONTHS))
        offers.append(generate_offer_path(HORIZON_MONTHS, rng))
    return SimInputs(population, shock_uniforms, offers, n)


def simulate_moments(theta: Tuple[float, float, float], sim: SimInputs,
                      base_params: Params) -> np.ndarray:
    alpha, cog_payday, shock_p_start = theta
    params = replace(base_params, alpha=alpha, cog_payday=cog_payday)
    cs = ChoiceSet(default_actions(params), params)
    pop = sim.population

    ever_captured = 0
    shortfall_months = 0
    payday_months = 0
    total_months = 0
    defer_months = 0

    for i in range(sim.n):
        hh = Household(pop["income"][i], pop["obligations"][i], pop["balance"][i],
                        pop["debt"][i], pop["alpha"][i], params)
        shocks_i = shock_path_from_uniforms(sim.shock_uniforms[i], shock_p_start, SHOCK_P_END)
        captured = False
        for t in range(HORIZON_MONTHS):
            income_t = (hh.income * (1 - SHOCK_SEVERITY) if shocks_i[t] else hh.income)
            ctx = hh.begin_period(t, income_t, 0.0, bool(sim.offers[i][t]))
            dec = cs.choose(hh, ctx)
            total_months += 1
            if ctx.resources < ctx.required:
                shortfall_months += 1
                if dec.chosen == PAYDAY:
                    payday_months += 1
            if dec.chosen == DEFER:
                defer_months += 1
            if dec.chosen in (PAYDAY, COLLAPSE):
                captured = True
            hh.apply_action(dec.chosen, ctx)
        ever_captured += captured

    m1 = ever_captured / sim.n
    m2 = (payday_months / shortfall_months) if shortfall_months > 0 else 0.0
    m3 = defer_months / total_months
    return np.array([m1, m2, m3])


def objective(theta, sim: SimInputs, base_params: Params, target: np.ndarray) -> float:
    sim_m = simulate_moments(theta, sim, base_params)
    return float(np.sum((sim_m - target) ** 2))


def run_smm(sim: SimInputs, base_params: Params, target: np.ndarray,
            bounds=((0.05, 2.0), (0.02, 0.90), (0.002, 0.10)), seed: int = 0,
            popsize: int = 10, maxiter: int = 25):
    result = differential_evolution(
        objective, bounds, args=(sim, base_params, target),
        strategy="best1bin", popsize=popsize, maxiter=maxiter,
        tol=1e-6, seed=seed, polish=True,
    )
    return result

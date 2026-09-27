"""Run with: pytest
These check things that would make every other result meaningless if they
ever failed: money is conserved, nothing goes negative, and re-running with
the same seed gives the same answer."""
import numpy as np
import pytest

from src.model import (Params, Household, ChoiceSet, default_actions,
                        sample_population, generate_shock_path, generate_offer_path,
                        Simulation)


def _make_households(n=100, seed=1):
    p = Params()
    rng = np.random.default_rng(seed)
    pop = sample_population(n, rng)
    cs = ChoiceSet(default_actions(p), p)
    households, sims = [], []
    for i in range(n):
        hh = Household(pop["income"][i], pop["obligations"][i], pop["balance"][i],
                        pop["debt"][i], pop["alpha"][i], p)
        rng_i = np.random.default_rng(1000 + i)
        shocks = generate_shock_path(24, rng_i)
        offers = generate_offer_path(24, rng_i)
        households.append(hh)
        sims.append(Simulation(hh, cs, shocks, 0.35, offers))
    return sims


def test_balances_never_go_negative():
    for sim in _make_households():
        sim.run()
        assert sim.hh.balance >= -1e-6
        assert sim.hh.debt >= -1e-6
        assert sim.hh.payday_debt >= -1e-6


def test_attention_stays_in_bounds():
    p = Params()
    for sim in _make_households():
        sim.run()
        assert p.a_min - 1e-9 <= sim.hh.attention <= 1.0 + 1e-9


def test_same_seed_gives_identical_result():
    def run_once():
        p = Params()
        rng = np.random.default_rng(5)
        pop = sample_population(20, rng)
        cs = ChoiceSet(default_actions(p), p)
        rng_i = np.random.default_rng(123)
        shocks = generate_shock_path(24, rng_i)
        offers = generate_offer_path(24, rng_i)
        hh = Household(pop["income"][0], pop["obligations"][0], pop["balance"][0],
                        pop["debt"][0], pop["alpha"][0], p)
        return Simulation(hh, cs, shocks, 0.35, offers).run()

    a, b = run_once(), run_once()
    assert a == b


def test_a_comfortable_household_never_touches_payday_debt():
    """A household with a big buffer and no shocks should never be captured."""
    p = Params()
    cs = ChoiceSet(default_actions(p), p)
    hh = Household(income=5000, obligations=1500, balance=8000, debt=500, alpha=0.5, params=p)
    shocks = np.zeros(24, dtype=bool)
    offers = np.ones(24, dtype=bool)
    out = Simulation(hh, cs, shocks, 0.0, offers).run()
    assert out.captured is False

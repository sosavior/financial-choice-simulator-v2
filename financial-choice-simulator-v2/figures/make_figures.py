"""
Generates every figure used in the paper, directly from the same model and
the same estimated parameters used elsewhere in this project. Run this
AFTER experiments/02_estimate_from_real_shed.py (or 01) has produced a
results/*.json file.

Usage:
    python figures/make_figures.py

Output: figures/fig1_policy_comparison.png
        figures/fig2_sensitivity_heatmap.png
        figures/fig3_example_trajectories.png
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (Params, Household, ChoiceSet, default_actions,
                        sample_population, generate_offer_path,
                        NoIntervention, LiquidityTransfer, sludge_reduced_params,
                        PAYDAY, COLLAPSE, DEFER, REFI)
from src.model.shocks import generate_shock_path
from src.estimation.stats import proportion_ci

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")


def load_params():
    for fname in ("estimated_params_real.json", "estimated_params.json"):
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            return Params(alpha=d["alpha"], cog_payday=d["cog_payday"]), d["shock_p_start"], fname
    return Params(), 0.02, "defaults (no results file found)"


BASE_PARAMS, SHOCK_P_START, SOURCE = load_params()
print(f"Using parameters from: {SOURCE}")
HORIZON = 12
SHOCK_SEVERITY = 0.35
N = 4000


# ---------------------------------------------------------------------------
# Figure 1: policy comparison bar chart with real 95% CIs
# ---------------------------------------------------------------------------
def run_cohort(params, intervention, pop, shocks, offers, n):
    cs = ChoiceSet(default_actions(params), params)
    captured = np.zeros(n, dtype=bool)
    for i in range(n):
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


def make_figure_1():
    rng = np.random.default_rng(42)
    pop = sample_population(N, rng)
    shocks = [generate_shock_path(HORIZON, np.random.default_rng(rng.integers(1_000_000_000)),
                                   p_start=SHOCK_P_START) for _ in range(N)]
    offers = [generate_offer_path(HORIZON, np.random.default_rng(rng.integers(1_000_000_000)))
              for _ in range(N)]
    sludge_params = sludge_reduced_params(BASE_PARAMS, 0.20)

    arms = {
        "Control": (BASE_PARAMS, NoIntervention()),
        "Cash\ntransfer": (BASE_PARAMS, LiquidityTransfer(500.0, 3)),
        "Sludge\nreduction": (sludge_params, NoIntervention()),
        "Both": (sludge_params, LiquidityTransfer(500.0, 3)),
    }
    rates, los, his = [], [], []
    for name, (params, iv) in arms.items():
        captured = run_cohort(params, iv, pop, shocks, offers, N)
        p = captured.mean()
        lo, hi = proportion_ci(p, N)
        rates.append(p); los.append(p - lo); his.append(hi - p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#8C92AC", "#4C72B0", "#55A868", "#8172B2"]
    x = np.arange(len(arms))
    ax.bar(x, rates, yerr=[los, his], capsize=5, color=colors, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(arms.keys())
    ax.set_ylabel("12-month payday-debt capture rate")
    ax.set_title("Policy comparison (paired simulation, n=4,000, 95% CI)")
    for i, r in enumerate(rates):
        ax.text(i, r + his[i] + 0.001, f"{r:.1%}", ha="center", fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_policy_comparison.png"), dpi=300)
    plt.close(fig)
    print("Wrote fig1_policy_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: sensitivity heatmap, capture rate vs. (alpha, cog_payday)
# ---------------------------------------------------------------------------
def make_figure_2():
    n_sens = 400
    rng = np.random.default_rng(11)
    pop = sample_population(n_sens, rng)
    shocks = [generate_shock_path(HORIZON, np.random.default_rng(rng.integers(1_000_000_000)),
                                   p_start=SHOCK_P_START) for _ in range(n_sens)]
    offers = [generate_offer_path(HORIZON, np.random.default_rng(rng.integers(1_000_000_000)))
              for _ in range(n_sens)]

    alphas = np.linspace(0.1, 2.0, 6)
    cogs = np.linspace(0.05, 0.85, 6)
    grid = np.zeros((len(alphas), len(cogs)))
    for i, a in enumerate(alphas):
        for j, c in enumerate(cogs):
            p = Params(alpha=a, cog_payday=c)
            captured = run_cohort(p, NoIntervention(), pop, shocks, offers, n_sens)
            grid[i, j] = captured.mean()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="rocket" if False else "viridis")
    ax.set_xticks(range(len(cogs))); ax.set_xticklabels([f"{c:.2f}" for c in cogs])
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("cognitive cost of predatory credit (cog_payday)")
    ax.set_ylabel("stress elasticity (alpha)")
    ax.set_title("Capture rate across the (alpha, cog_payday) grid\n"
                  "(control arm, no policy, n=400 per cell)")
    for i in range(len(alphas)):
        for j in range(len(cogs)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                     color="white" if grid[i, j] > grid.max() * 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="capture rate")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_sensitivity_heatmap.png"), dpi=300)
    plt.close(fig)
    print("Wrote fig2_sensitivity_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3: example household trajectories (bandwidth over time)
# ---------------------------------------------------------------------------
def make_figure_3():
    rng = np.random.default_rng(5)
    pop = sample_population(30, rng)
    cs = ChoiceSet(default_actions(BASE_PARAMS), BASE_PARAMS)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    picks = [0, 1, 2]
    for ax, i in zip(axes, picks):
        hh = Household(pop["income"][i], pop["obligations"][i], pop["balance"][i],
                        pop["debt"][i], pop["alpha"][i], BASE_PARAMS)
        rng_i = np.random.default_rng(100 + i)
        shocks = generate_shock_path(HORIZON, rng_i, p_start=max(SHOCK_P_START, 0.15))
        offers = generate_offer_path(HORIZON, rng_i)
        attn, actions = [], []
        for t in range(HORIZON):
            income_t = hh.income * (1 - SHOCK_SEVERITY) if shocks[t] else hh.income
            ctx = hh.begin_period(t, income_t, 0.0, bool(offers[t]))
            dec = cs.choose(hh, ctx)
            hh.apply_action(dec.chosen, ctx)
            attn.append(hh.attention); actions.append(dec.chosen)
        months = np.arange(1, HORIZON + 1)
        ax.plot(months, attn, "-o", color="#4C72B0", markersize=4)
        ax.axhline(BASE_PARAMS.cog_refi, color="grey", lw=0.8, ls="--", label="cog(refi)")
        ax.axhline(BASE_PARAMS.cog_defer, color="grey", lw=0.8, ls=":", label="cog(defer)")
        for t, s in enumerate(shocks):
            if s:
                ax.axvspan(t + 0.5, t + 1.5, color="red", alpha=0.08)
        ax.set_title(f"Household {i+1}", fontsize=10)
        ax.set_xlabel("month")
    axes[0].set_ylabel("cognitive bandwidth (attention)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Example bandwidth trajectories (shaded = income-shock month)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_example_trajectories.png"), dpi=300)
    plt.close(fig)
    print("Wrote fig3_example_trajectories.png")


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
    print("\nAll figures written to the figures/ folder.")

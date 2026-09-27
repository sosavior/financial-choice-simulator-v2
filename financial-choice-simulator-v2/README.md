# Financial Choice Simulator

A structural, agent-based simulation of household financial choice under
liquidity and cognitive-bandwidth constraints, calibrated with the
Simulated Method of Moments (SMM) against **real, cited Federal Reserve
SHED statistics** — and a full LaTeX paper reporting what it actually finds.

This is a from-scratch rebuild. Nothing here is fabricated: every number
either comes out of a script you can rerun, or is a real published Federal
Reserve statistic with a source URL attached to it in the code.

## What's real vs. what's assumed

| Piece | Status |
|---|---|
| Household bandwidth/choice mechanics (`src/model/`) | A model — a stated set of assumptions, not measured |
| Interest rates, fees (`src/model/params.py`) | ASSUMED, labeled as such in the file |
| Income/debt/balance distribution (`src/model/population.py`) | SYNTHETIC, labeled as such |
| `alpha` (stress elasticity) | ESTIMATED via SMM |
| `cog_payday` (cognitive cost of predatory credit) | ESTIMATED via SMM |
| `shock_p_start` (monthly income-shock probability) | ESTIMATED via SMM |
| The 3 target moments used to estimate the above | **REAL**, cited 2025 Federal Reserve SHED codebook counts (see `src/data/shed_real_moments.py`) |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                     # 1. sanity checks
python experiments/01_validate_estimator.py   # 2. does the estimator work? (synthetic ground truth)
python experiments/02_estimate_from_real_shed.py  # 3. calibrate against REAL Fed data
python experiments/03_policy_counterfactual.py    # 4. the actual policy comparison
python figures/make_figures.py                    # 5. regenerate the paper's figures
```

or just `python run_all.py` to do 1–4 in order.

## Project layout

```
src/model/          the simulation engine (Household, ChoiceSet, Params, Simulation)
src/estimation/      the SMM estimator, bootstrap, and statistics helpers
src/data/            real, cited Federal Reserve SHED target moments
experiments/         the three scripts above, in run order
tests/               pytest invariant checks (money conserved, bounds respected, reproducibility)
figures/             figure-generation script + output PNGs used in the paper
results/             JSON output of the estimation scripts (gitignored by default; regenerate anytime)
paper/               the LaTeX paper (paper.tex) reporting these exact results
```

## The real Federal Reserve data being used right now

Three real, cited counts from the **2025 Survey of Household Economics and
Decisionmaking (SHED)** codebook (n = 12,934, fielded October 2025, released
May 13, 2026):

- `BK2_c` — took out a payday loan/advance in the past 12 months: 501/12,934 = 3.87%
- `C4A` — carried an unpaid card balance "most or all of the time" (cardholders): 2,320/10,759 = 21.56%
- `C3P` — paid less than the minimum on a card last month (cardholders w/ a balance): 284/10,120 = 2.81%

Source: `https://www.federalreserve.gov/consumerscommunities/files/SHED_2025codebook.pdf`

**These are unweighted counts** taken directly from the published codebook
tabulations, not the survey's weighted population estimates. This is a
disclosed, real limitation — see "Known limitations" below, and "Adding the
full raw microdata" if you want to go further.

## Adding the full raw microdata (optional, more work, more precision)

The three moments above are enough to run the whole pipeline as-is. If you
want household-level heterogeneity (real income/debt distributions instead
of synthetic ones, and properly *weighted* moments), you can download the
full public-use file yourself — I cannot fetch it into this environment,
since federalreserve.gov isn't reachable from here:

1. Download the CSV from:
   `https://www.federalreserve.gov/consumerscommunities/files/SHED_public_use_data_2025_(CSV).zip`
2. Unzip it into a `data/` folder at the project root (not tracked by git — see `.gitignore`).
3. Before writing any mapping code, **validate the columns actually exist and mean what you expect**:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('data/public2025.csv', low_memory=False)
   print(df['BK2_c'].value_counts(dropna=False))
   print(df['weight'].describe())
   "
   ```
4. Recompute the three moments **weighted**: `(df['weight'] * (df['BK2_c']==1)).sum() / df['weight'].sum()`,
   and update `src/data/shed_real_moments.py` with the weighted figures (keep the unweighted
   ones too, for comparison — don't silently replace them).
5. If a column doesn't exist or the codebook has changed the coding since this was written,
   **stop and fix the mapping** rather than falling back to a default value. This exact
   failure mode (silent fallbacks producing fake-looking "real" data) is what went wrong in
   an earlier version of this project — see the paper's Section 2 for the full story.

## Known limitations (disclosed on purpose)

- **Unweighted counts.** See above. Get the raw CSV and use `weight` for a properly
  population-representative estimate.
- **m3 (deferred-payment share) fits poorly.** At the fitted parameters, the model rarely
  uses its DEFER action, underpredicting the real C3P figure by roughly 2.7 percentage
  points. This is a real, disclosed shortcoming of the current choice-set specification,
  not hidden in the paper's Results section.
- **The cash-transfer arm is a fixed-calendar-month transfer** (delivered in month 3
  regardless of when a household's income shock occurs), not a shock-triggered one. Given
  how infrequent shocks are at the fitted `shock_p_start` (~0.8%/month), most households
  never experience a shock anywhere near month 3, which likely understates how effective a
  *well-timed* cash transfer could be. A state-triggered transfer (delivered when the shock
  actually hits, whichever month that is) is a natural next step — `LiquidityTransfer` would
  need a triggered variant analogous to what's described in the paper's Future Work section.
- **Bootstrap standard errors are wide** (see `results/*.json`), reflecting a genuinely
  weakly-identified `alpha` at this population size / optimizer budget — increase
  `N_ESTIMATION` and the `popsize`/`maxiter` arguments to `run_smm()` for tighter estimates,
  at the cost of runtime.

## Reproducing the paper exactly

```
pytest -q
python experiments/01_validate_estimator.py
python experiments/02_estimate_from_real_shed.py
python experiments/03_policy_counterfactual.py
python figures/make_figures.py
```

Then compile `paper/paper.tex` (e.g. on Overleaf). Every number in the paper
traces back to the console output of one of these five commands.

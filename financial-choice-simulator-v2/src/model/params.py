"""
All model parameters live here, and ONLY here. Nothing elsewhere in this
project should hard-code a number that belongs in this file.

Every field below is labeled as either:

  ASSUMED    -- a plausible, stated modeling choice. Not measured from data.
               If you later plug in real survey data, these are the values
               you should look to replace or justify first.

  ESTIMATED  -- recovered by the SMM estimator in src/estimation/smm.py.
               The values given here are just starting guesses used before
               estimation runs; run_estimation.py overwrites them.

All interest rates are MONTHLY. All dollar amounts are monthly dollars.
"""
from dataclasses import dataclass


PAY_MIN = "pay_min"
REFI = "refinance"
DEFER = "defer_payment"
PAYDAY = "payday_loan"
COLLAPSE = "cognitive_collapse"  # no option was accessible; forced worst case


@dataclass(frozen=True)
class Params:
    # ---- interest rates (ASSUMED) --------------------------------------
    r_formal_high: float = 0.018     # ~21.6% APR, un-refinanced revolving rate
    r_formal_low: float = 0.009      # ~10.8% APR, rate after successful refi
    r_payday: float = 0.18           # ~216% APR-equivalent monthly compounding

    # ---- fees (ASSUMED) --------------------------------------------------
    payday_fee: float = 0.20         # fee as a fraction of amount borrowed
    refi_fee: float = 150.0          # flat dollar fee to refinance
    defer_fee: float = 35.0          # flat dollar fee to defer/partial-pay
    late_fee_rate: float = 0.10      # fraction of any unpaid gap

    min_pay_rate: float = 0.03       # minimum payment = 3% of balance

    # ---- cognitive ("sludge") cost of EACH action, on a 0-1 scale --------
    # Higher = harder to access when bandwidth is depleted.
    # Ordering matters more than exact values: formal help should be the
    # hardest thing to access, predatory credit the easiest. (ASSUMED)
    cog_pay_min: float = 0.05
    cog_payday: float = 0.15         # ESTIMATED by src/estimation/smm.py
    cog_defer: float = 0.45
    cog_refi: float = 0.75

    # ---- cognitive bandwidth dynamics -------------------------------------
    alpha: float = 0.50              # ESTIMATED: stress -> bandwidth elasticity
    a_min: float = 0.02              # bandwidth floor (never hits exactly 0)
    rho: float = 0.40                # ASSUMED: bandwidth adjustment speed (EMA)
    stress_cap: float = 3.0          # ASSUMED: cap on the stress ratio

    # ---- other -------------------------------------------------------------
    buffer_months: float = 0.5       # ASSUMED: months of obligations kept as buffer
    payday_cap_months: float = 1.0   # ASSUMED: max payday borrowing = 1 month income
    horizon: int = 12                # months, used only in cost-projection math

    def __post_init__(self):
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1]")
        if self.r_formal_low > self.r_formal_high:
            raise ValueError("refinancing must not raise the rate")
        for name in ("cog_pay_min", "cog_payday", "cog_defer", "cog_refi"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")

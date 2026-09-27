"""A single household: its financial state, and how that state evolves
when it takes an action in a given month."""
from dataclasses import dataclass
from typing import Optional
from .params import Params, PAY_MIN, REFI, DEFER, PAYDAY, COLLAPSE

EPS = 1e-9


@dataclass
class PeriodContext:
    """Everything computed at the start of a month, before a choice is made."""
    t: int
    income: float
    transfer: float
    resources: float     # cash on hand this month (balance + income - obligations)
    required: float       # minimum payment obligation this month
    pay_formal: float
    pay_payday: float
    stress: float
    refi_offer: bool = True


class Household:
    def __init__(self, income: float, obligations: float, balance: float,
                 debt: float, alpha: float, params: Optional[Params] = None):
        self.p = params or Params()
        self.income = float(income)
        self.obligations = float(obligations)
        self.balance = float(balance)
        self.debt = float(debt)
        self.payday_debt = 0.0
        self.alpha = float(alpha)
        self.rate = self.p.r_formal_high
        self.refinanced = False
        self.attention = 1.0  # cognitive bandwidth, 0-1, starts full

    @property
    def net_worth(self) -> float:
        return self.balance - self.debt - self.payday_debt

    def begin_period(self, t: int, income_t: float, transfer: float = 0.0,
                      refi_offer: bool = True) -> PeriodContext:
        p = self.p
        self.balance += transfer
        pay_formal = p.min_pay_rate * self.debt * (1.0 + self.rate)
        pay_payday = p.r_payday * self.payday_debt
        required = pay_formal + pay_payday
        resources = self.balance + income_t - self.obligations

        stress = min((self.obligations + required) / max(self.balance + income_t, 1.0),
                      p.stress_cap)
        target = max(p.a_min, 1.0 - self.alpha * stress)
        self.attention = min(1.0, max(p.a_min, (1 - p.rho) * self.attention + p.rho * target))

        return PeriodContext(t, income_t, transfer, resources, required,
                              pay_formal, pay_payday, stress, bool(refi_offer))

    def payday_split(self, ctx: PeriodContext):
        p = self.p
        shortfall = max(0.0, ctx.required - ctx.resources)
        room = max(0.0, p.payday_cap_months * self.income - self.payday_debt)
        loan = min(shortfall, room / (1.0 + p.payday_fee))
        return loan, shortfall - loan

    def apply_action(self, name: str, ctx: PeriodContext) -> dict:
        p = self.p
        X = ctx.resources
        interest = self.debt * self.rate + self.payday_debt * p.r_payday
        fees, loan, gap = 0.0, 0.0, 0.0
        new_debt = self.debt
        new_payday = self.payday_debt

        if name == PAY_MIN:
            new_balance = X - ctx.required
            new_debt = self.debt * (1 + self.rate) - ctx.pay_formal
            new_payday = self.payday_debt * (1 + p.r_payday) - ctx.pay_payday

        elif name == REFI:
            fees = p.refi_fee
            new_balance = X - ctx.required - fees
            new_debt = self.debt * (1 + self.rate) - ctx.pay_formal
            new_payday = self.payday_debt * (1 + p.r_payday) - ctx.pay_payday
            self.rate = p.r_formal_low
            self.refinanced = True

        elif name == DEFER:
            fees = p.defer_fee
            new_balance = max(0.0, X) - ctx.pay_payday
            new_debt = self.debt * (1 + self.rate) - ctx.pay_formal + fees
            new_payday = self.payday_debt * (1 + p.r_payday) - ctx.pay_payday

        elif name in (PAYDAY, COLLAPSE):
            # COLLAPSE (nothing was accessible) is treated financially the
            # same as a forced payday loan -- the worst-case outcome.
            loan, gap = self.payday_split(ctx)
            fees = p.payday_fee * loan + p.late_fee_rate * gap
            new_balance = 0.0
            new_debt = self.debt * (1 + self.rate) - ctx.pay_formal + gap * (1 + p.late_fee_rate)
            new_payday = (self.payday_debt * (1 + p.r_payday) - ctx.pay_payday
                          + loan * (1 + p.payday_fee))
        else:
            raise ValueError(f"unknown action {name!r}")

        new_balance = max(0.0, new_balance)
        buffer = p.buffer_months * self.obligations
        excess = new_balance - buffer
        if excess > 0:
            pay_pd = min(new_payday, excess)
            new_payday -= pay_pd
            pay_dt = min(new_debt, excess - pay_pd)
            new_debt -= pay_dt
            new_balance -= (pay_pd + pay_dt)

        self.balance = new_balance
        self.debt = max(0.0, new_debt)
        self.payday_debt = max(0.0, new_payday)
        return {"interest": interest, "fees": fees, "loan": loan, "gap": gap}

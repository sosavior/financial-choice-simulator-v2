"""Defines which actions a household CAN take (feasibility), which of those
it is COGNITIVELY ABLE to take (accessibility), and which one it picks."""
from dataclasses import dataclass
from typing import List, Optional
from .household import Household, PeriodContext
from .params import Params, PAY_MIN, REFI, DEFER, PAYDAY, COLLAPSE

EPS = 1e-9


@dataclass(frozen=True)
class Action:
    name: str
    cognitive_cost: float


@dataclass
class Decision:
    chosen: str
    feasible: List[str]
    accessible: List[str]
    collapsed: bool  # True if no feasible action was cognitively accessible


def default_actions(p: Optional[Params] = None) -> List[Action]:
    p = p or Params()
    return [
        Action(PAY_MIN, p.cog_pay_min),
        Action(REFI, p.cog_refi),
        Action(DEFER, p.cog_defer),
        Action(PAYDAY, p.cog_payday),
    ]


class ChoiceSet:
    def __init__(self, actions: List[Action], params: Optional[Params] = None):
        self.actions = list(actions)
        self.p = params or Params()

    def is_feasible(self, a: Action, hh: Household, ctx: PeriodContext) -> bool:
        X, R = ctx.resources, ctx.required
        if a.name == PAY_MIN:
            return X >= R - EPS
        if a.name == REFI:
            return ctx.refi_offer and (not hh.refinanced) and X >= R + self.p.refi_fee - EPS
        if a.name == DEFER:
            return 0.0 - EPS <= X < R - EPS
        if a.name == PAYDAY:
            return X < R - EPS
        raise ValueError(a.name)

    def expected_cost(self, a: Action, hh: Household, ctx: PeriodContext) -> float:
        """Long-run cost estimate used to rank feasible+accessible options.
        Lower is better. PAY_MIN is the reference (cost 0)."""
        p, H = self.p, self.p.horizon
        if a.name == PAY_MIN:
            return 0.0
        if a.name == REFI:
            remaining_debt = hh.debt * (1 + hh.rate) - ctx.pay_formal
            savings = H * remaining_debt * (hh.rate - p.r_formal_low)
            return p.refi_fee - savings
        if a.name == DEFER:
            return p.defer_fee + H * hh.rate * ctx.pay_formal
        if a.name == PAYDAY:
            loan, gap = hh.payday_split(ctx)
            return (loan * (p.payday_fee + p.r_payday * H)
                    + gap * (p.late_fee_rate + hh.rate * H))
        raise ValueError(a.name)

    def choose(self, hh: Household, ctx: PeriodContext) -> Decision:
        feasible = [a for a in self.actions if self.is_feasible(a, hh, ctx)]
        accessible = [a for a in feasible if a.cognitive_cost <= hh.attention + EPS]

        if not accessible:
            return Decision(COLLAPSE, [a.name for a in feasible], [], collapsed=True)

        costs = {a.name: self.expected_cost(a, hh, ctx) for a in accessible}
        best = min(accessible, key=lambda a: (costs[a.name], a.cognitive_cost, a.name))
        return Decision(best.name, [a.name for a in feasible],
                         [a.name for a in accessible], collapsed=False)

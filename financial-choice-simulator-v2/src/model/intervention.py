"""Policy levers you can apply to a cohort before simulating it."""
from dataclasses import dataclass, replace
from typing import Optional
from .params import Params


class Intervention:
    def transfer(self, t: int, household) -> float:
        return 0.0


class NoIntervention(Intervention):
    pass


@dataclass
class LiquidityTransfer(Intervention):
    """A one-time cash transfer delivered in a specific month."""
    amount: float
    period: int

    def transfer(self, t: int, household) -> float:
        return self.amount if t == self.period else 0.0


def sludge_reduced_params(base: Params, new_cog_refi: float) -> Params:
    """Returns a copy of `base` with formal refinancing made easier to
    access (lower cognitive cost). Use this to simulate a 'reduce the
    paperwork burden' policy."""
    return replace(base, cog_refi=new_cog_refi)

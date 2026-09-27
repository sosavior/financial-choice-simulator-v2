"""Runs a single household through a fixed number of months and records
what happened. This is the one and only simulation loop in the project --
every script (estimation, bootstrap, policy comparison) calls this."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .household import Household
from .choice_set import ChoiceSet
from .intervention import Intervention, NoIntervention
from .params import PAYDAY, COLLAPSE


@dataclass
class Outcome:
    captured: bool             # ever forced into payday debt or collapse
    months_captured: int
    ever_refinanced: bool
    final_net_worth: float
    final_payday_debt: float


class Simulation:
    def __init__(self, household: Household, choice_set: ChoiceSet,
                 shock_path: np.ndarray, shock_severity: float = 0.35,
                 offer_path: Optional[np.ndarray] = None,
                 intervention: Optional[Intervention] = None):
        self.hh = household
        self.cs = choice_set
        self.shock_path = shock_path
        self.shock_severity = shock_severity
        self.T = len(shock_path)
        self.offer_path = (np.ones(self.T, dtype=bool) if offer_path is None
                            else offer_path)
        self.intervention = intervention or NoIntervention()

    def run(self) -> Outcome:
        hh = self.hh
        months_captured = 0
        for t in range(self.T):
            income_t = hh.income * (1 - self.shock_severity) if self.shock_path[t] else hh.income
            transfer = self.intervention.transfer(t, hh)
            ctx = hh.begin_period(t, income_t, transfer, bool(self.offer_path[t]))
            decision = self.cs.choose(hh, ctx)
            hh.apply_action(decision.chosen, ctx)
            if decision.chosen in (PAYDAY, COLLAPSE):
                months_captured += 1

        return Outcome(
            captured=months_captured > 0,
            months_captured=months_captured,
            ever_refinanced=hh.refinanced,
            final_net_worth=hh.net_worth,
            final_payday_debt=hh.payday_debt,
        )

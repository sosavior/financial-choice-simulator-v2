"""
Real Federal Reserve SHED-derived target moments.

Every number below is computed from raw response COUNTS taken directly from
the Board of Governors' own published codebook for the 2025 Survey of
Household Economics and Decisionmaking (SHED):

    Source:  Codebook for the 2025 Survey of Household Economics and
             Decisionmaking (PDF), Board of Governors of the Federal
             Reserve System.
    URL:     https://www.federalreserve.gov/consumerscommunities/files/SHED_2025codebook.pdf
    Sample:  n = 12,934 adults, fielded October 2025, released May 13, 2026.
    Retrieved: verify the current codebook still shows these counts before
               citing this in anything you publish -- codebooks are
               occasionally revised (this one was, twice, in 2026).

These are UNWEIGHTED response counts, not the survey-weighted population
estimates the Fed itself reports in "Economic Well-Being of U.S. Households
in 2025." If you obtain the raw public-use CSV (see README), recompute these
weighted (using the "weight" column) before treating them as nationally
representative -- unweighted counts from a stratified sample can differ
from the weighted population share, sometimes by several points.

Two moments are used, chosen to match the two behaviors the structural
model actually represents:

  m1 = share of respondents who took out a payday loan or payday advance
       in the past 12 months (variable BK2_c)
       -> matches the model's PAYDAY action

  m2 = among credit-card holders, share who carried an unpaid balance
       "most or all of the time" in the past 12 months (variable C4A == 3)
       -> matches the model's chronic PAY_MIN / revolving-debt state

The model's simulation horizon (HORIZON_MONTHS in src/estimation/smm.py) is
set to 12 specifically so it asks about the same recall window as these two
survey questions.
"""
from dataclasses import dataclass
import numpy as np

CODEBOOK_URL = "https://www.federalreserve.gov/consumerscommunities/files/SHED_2025codebook.pdf"
CODEBOOK_SAMPLE_N = 12_934


@dataclass(frozen=True)
class CitedCount:
    variable: str
    question: str
    numerator: int
    denominator: int
    note: str

    @property
    def share(self) -> float:
        return self.numerator / self.denominator


# BK2_c: "Take out a payday loan or payday advance - In the past 12 months,
# did you (and/or your spouse or partner):"  Yes = 501, No = 12,433, n=12,934
PAYDAY_LOAN_12MO = CitedCount(
    variable="BK2_c",
    question="Took out a payday loan or payday advance in the past 12 months",
    numerator=501,
    denominator=12_934,
    note="Unweighted, whole sample (question has no skip pattern / no missing).",
)

# BK2_d: pawn shop / auto title loan, same structure -- kept as a secondary,
# broader "high-cost alternative credit" figure for robustness discussion,
# NOT used as a primary SMM target (to avoid double-counting households that
# appear in both BK2_c and BK2_d, which would require the raw microdata to
# resolve correctly).
PAWN_AUTO_TITLE_LOAN_12MO = CitedCount(
    variable="BK2_d",
    question="Took out a pawn shop loan or auto title loan in the past 12 months",
    numerator=299,
    denominator=12_934,
    note="Unweighted. Reported for robustness discussion only; see docstring.",
)

# C4A: "In the past 12 months, how frequently have you carried an unpaid
# balance on one or more of your credit cards?" asked of the 10,759
# respondents who hold a credit card (C2A == Yes). Value 3 = "Most or all
# of the time": 2,320 of 10,759 cardholders.
CHRONIC_REVOLVER_SHARE = CitedCount(
    variable="C4A",
    question="Carried an unpaid credit-card balance 'most or all of the time' "
              "in the past 12 months, among credit-card holders",
    numerator=2_320,
    denominator=10_759,
    note="Denominator restricted to the 10,759 respondents with C2A == Yes "
         "(i.e. who hold at least one credit card); unweighted.",
)

# C3P, kept for robustness/reference only (see README: this one is closer to
# a snapshot of "last month" rather than a 12-month frequency, so it is a
# noisier match to the model's chronic-state moment than C4A).
MISSED_MIN_PAYMENT_LAST_MONTH = CitedCount(
    variable="C3P",
    question="Did not pay, or paid less than, the minimum payment on at "
              "least one credit card last month, among cardholders with a "
              "balance",
    numerator=284,
    denominator=284 + 9_836,
    note="Denominator excludes the 639 respondents who carried no card "
         "balance at all last month (C3P == -9).",
)

# A0 / A1_a: credit application rejection rate, conditional on applying --
# reported for the paper's robustness/discussion section, not used as an
# SMM target (the structural model doesn't currently have an explicit
# 'formal application' stage separate from the refinance action).
CREDIT_REJECTION_RATE = CitedCount(
    variable="A1_a (conditional on A0 == Yes)",
    question="Turned down for credit, among those who applied for credit "
              "in the past 12 months",
    numerator=1_065,
    denominator=4_276,
    note="Unweighted. A0B (desired credit but did not apply) is a "
         "DIFFERENT, self-censorship measure and should not be summed with "
         "this -- a mistake the previous version of this project made.",
)


def real_target_vector() -> np.ndarray:
    """The three moments actually used to calibrate the model, in the same
    [m1, m2, m3] order src/estimation/smm.py expects."""
    return np.array([
        PAYDAY_LOAN_12MO.share,
        CHRONIC_REVOLVER_SHARE.share,
        MISSED_MIN_PAYMENT_LAST_MONTH.share,
    ])


def print_provenance():
    for c in (PAYDAY_LOAN_12MO, PAWN_AUTO_TITLE_LOAN_12MO,
              CHRONIC_REVOLVER_SHARE, MISSED_MIN_PAYMENT_LAST_MONTH,
              CREDIT_REJECTION_RATE):
        print(f"{c.variable:28s} {c.numerator:>6,}/{c.denominator:<6,} = {c.share:.4f}   {c.question}")
    print(f"\nSource: {CODEBOOK_URL}  (n={CODEBOOK_SAMPLE_N:,})")


if __name__ == "__main__":
    print_provenance()

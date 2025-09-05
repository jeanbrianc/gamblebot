"""Modeling utilities using a Poisson distribution."""
from __future__ import annotations

import math
import pandas as pd


def prob_two_plus(lmbda: float) -> float:
    """Probability of at least two touchdowns given rate λ."""
    return 1 - math.exp(-lmbda) * (1 + lmbda)


def add_model_probability(td_rate: pd.DataFrame) -> pd.DataFrame:
    """Add Poisson two+ TD probability to dataframe."""
    td_rate = td_rate.copy()
    td_rate["model_prob"] = td_rate["lambda"].apply(prob_two_plus)
    return td_rate

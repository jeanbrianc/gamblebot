"""Staking utilities using the Kelly criterion."""
from __future__ import annotations

import pandas as pd

from .odds import american_to_decimal


def add_edge_and_stake(
    df: pd.DataFrame,
    *,
    kelly_fraction: float = 0.5,
    unit_size: float = 1.0,
) -> pd.DataFrame:
    """Add edge and Kelly stake columns."""
    df = df.copy()
    df["edge"] = df["model_prob"] - df["implied_prob"]
    decimal_odds = df["odds"].apply(american_to_decimal)
    df["kelly"] = (
        (df["model_prob"] * (decimal_odds - 1) - (1 - df["model_prob"]))
        / (decimal_odds - 1)
    )
    df["kelly"] = df["kelly"].clip(lower=0)
    df["stake_units"] = df["kelly"] * kelly_fraction * unit_size
    return df.drop(columns=["kelly"])

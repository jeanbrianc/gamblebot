"""Simple probabilistic models for touchdown and sack props."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_model_probability(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: player, team, position, games, two_plus, mean_td, recent_opps
    Output columns: player, team, position, recent_opps, model_prob
    """
    df = features_df.copy()

    # League prior for P(X>=2)
    total_two_plus = pd.to_numeric(df["two_plus"], errors="coerce").fillna(0).sum()
    total_games = pd.to_numeric(df["games"], errors="coerce").fillna(0).sum()
    p0 = float(total_two_plus) / float(total_games) if total_games > 0 else 0.01
    p0 = float(np.clip(p0, 0.002, 0.05))  # plausible league prior

    # Poisson probability of >=2 given per-game mean λ
    lam = pd.to_numeric(df["mean_td"], errors="coerce").fillna(0.0).clip(lower=0.0)
    poisson_p_ge2 = 1.0 - np.exp(-lam) * (1.0 + lam)

    # Empirical-Bayes shrinkage
    prior_strength = 8.0  # pseudo-games
    g = pd.to_numeric(df["games"], errors="coerce").fillna(0.0)
    w = g / (g + prior_strength)
    model_prob = w * poisson_p_ge2 + (1.0 - w) * p0

    df["model_prob"] = np.clip(model_prob, 0.0, 0.30)

    return df[["player", "team", "position", "recent_opps", "model_prob"]]


def add_sack_model_probability(features_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate P(≥1 sack) for defenders using a Poisson model."""
    df = features_df.copy()

    pressure_rate = (
        pd.to_numeric(df["pressures_per_game"], errors="coerce").fillna(0.0)
        / pd.to_numeric(df["pass_rush_snaps"], errors="coerce").fillna(1.0)
    )
    prwin = pd.to_numeric(df["prwin"], errors="coerce").fillna(0.0)
    opp_drop = pd.to_numeric(df["opp_dropbacks"], errors="coerce").fillna(30.0)
    opp_sack_rate = pd.to_numeric(df["opp_sack_rate_allowed"], errors="coerce").fillna(0.07)

    lam = pressure_rate * prwin * opp_drop * opp_sack_rate
    poisson_p_ge1 = 1.0 - np.exp(-lam.clip(lower=0.0))

    # Empirical-Bayes shrinkage toward league average
    lam0 = float(lam.mean()) if len(lam) else 0.05
    p0 = 1.0 - np.exp(-lam0)
    snaps = pd.to_numeric(df["pass_rush_snaps"], errors="coerce").fillna(0.0)
    prior_strength = 200.0  # pseudo-snaps
    w = snaps / (snaps + prior_strength)
    model_prob = w * poisson_p_ge1 + (1.0 - w) * p0

    df["model_prob"] = np.clip(model_prob, 0.0, 0.95)
    return df[["player", "team", "opponent", "position", "pass_rush_snaps", "model_prob"]]


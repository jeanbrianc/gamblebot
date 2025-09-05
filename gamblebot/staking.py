"""Staking utilities using the Kelly criterion."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .odds import american_to_decimal


# Common names we’ll try for your model probability column
_PROB_COL_CANDIDATES: tuple[str, ...] = (
    "model_prob",
    "p_model",
    "probability",
    "prob",
    "p",
    "p_two_td",
)


def _find_prob_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # Heuristic: first float col that looks like probabilities in (0,1)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            s = df[c].dropna()
            if not s.empty and (s.between(0.0, 1.0).mean() > 0.9):
                return c
    raise KeyError(
        f"Could not find a model probability column. "
        f"Tried {list(candidates)}; available columns: {list(df.columns)}"
    )


def _ensure_decimal_odds(df: pd.DataFrame) -> pd.Series:
    """
    Return a decimal-odds Series from one of:
      - 'decimal' already present
      - 'american' or legacy 'odds' (American) -> convert
    """
    if "decimal" in df.columns:
        dec = pd.to_numeric(df["decimal"], errors="coerce")
        if dec.notna().any():
            return dec

    american_col = "american" if "american" in df.columns else ("odds" if "odds" in df.columns else None)
    if american_col is not None:
        american = pd.to_numeric(df[american_col], errors="coerce")
        return american.apply(american_to_decimal)

    raise KeyError(
        "No odds column found. Expected one of 'decimal', 'american', or 'odds'. "
        f"Available columns: {list(df.columns)}"
    )


def add_edge_and_stake(
    df: pd.DataFrame,
    *,
    kelly_fraction: float = 0.5,
    unit_size: float = 1.0,
    prob_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Adds:
      - decimal (if missing), implied_prob (1/decimal if missing)
      - edge = model_prob - implied_prob
      - kelly_full = (b*p - (1-p)) / b, where b = decimal-1
      - stake_units = max(0, kelly_fraction * kelly_full)
      - stake_amount = stake_units * unit_size
    """
    out = df.copy()

    # Decimal odds & implied prob
    out["decimal"] = _ensure_decimal_odds(out)
    if "implied_prob" not in out.columns or out["implied_prob"].isna().all():
        out["implied_prob"] = np.where(out["decimal"] > 1.0, 1.0 / out["decimal"], np.nan)

    # Model probability
    p_col = prob_col or _find_prob_col(out, _PROB_COL_CANDIDATES)
    p = out[p_col].astype(float).clip(1e-9, 1 - 1e-9)

    # Kelly
    b = (out["decimal"] - 1.0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kelly_full = (b * p - (1.0 - p)) / b
    kelly_full = kelly_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Edge & stakes
    out["edge"] = (p - out["implied_prob"]).astype(float)
    out["kelly_full"] = kelly_full
    out["stake_units"] = (kelly_fraction * kelly_full).clip(lower=0.0)
    out["stake_amount"] = out["stake_units"] * float(unit_size)

    # Nice column order
    front = [c for c in ["player", "book", "american", "decimal", "implied_prob", p_col, "edge", "stake_units", "stake_amount"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


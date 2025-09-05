"""Filtering utilities for player reports."""
from __future__ import annotations

import pandas as pd
from typing import Iterable


def _normalize_positions(pos: Iterable[str]) -> list[str]:
    return [p.strip().upper() for p in pos if p and isinstance(p, str)]


def apply_filters(
    df: pd.DataFrame,
    *,
    positions: list[str],
    min_recent_opps: float,
    exclude_injured: bool,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Filter offensive players for the 2+ TD report.

    Parameters mirror those in the CLI. Injury filtering is a stub that currently
    acts as a no-op but preserves the interface for future enhancement.
    """
    out = df.copy()
    if positions:
        pos_set = set(_normalize_positions(positions))
        if "position" in out.columns:
            out = out[out["position"].str.upper().isin(pos_set)]
    if min_recent_opps is not None and "recent_opps" in out.columns:
        out = out[out["recent_opps"] >= float(min_recent_opps)]
    # Injury filtering could be added here in the future. For now this is a no-op.
    return out.reset_index(drop=True)


def apply_passrush_filters(
    df: pd.DataFrame,
    *,
    positions: list[str],
    min_passrush_snaps: float,
    exclude_injured: bool,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Filter defensive players for the sacks report.

    Currently filters by position and pass-rush snap volume. Injury filtering is
    a placeholder for future integration with injury reports.
    """
    out = df.copy()
    if positions:
        pos_set = set(_normalize_positions(positions))
        if "position" in out.columns:
            out = out[out["position"].str.upper().isin(pos_set)]
    if min_passrush_snaps is not None and "pass_rush_snaps" in out.columns:
        out = out[out["pass_rush_snaps"] >= float(min_passrush_snaps)]
    return out.reset_index(drop=True)

"""Feature engineering for player touchdown rates."""
from __future__ import annotations

import pandas as pd


def compute_td_rate(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute a player's historical touchdown rate (λ).

    Parameters
    ----------
    weekly: pandas.DataFrame
        Weekly player stats from :func:`nfl_data_py.import_weekly_data`.
    """
    # touchdowns per game
    weekly = weekly.copy()
    weekly["td"] = weekly["rushing_td"].fillna(0) + weekly["receiving_td"].fillna(0)
    grouped = (
        weekly.groupby(["player_id", "player_display_name", "recent_team"], as_index=False)
        .agg(td=("td", "sum"), games=("week", "nunique"))
    )
    grouped["lambda"] = grouped["td"] / grouped["games"].clip(lower=1)
    return grouped.rename(
        columns={
            "player_display_name": "player",
            "recent_team": "team",
        }
    )[["player_id", "player", "team", "lambda"]]

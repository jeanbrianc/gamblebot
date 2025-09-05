"""Feature engineering for 2+ TD modeling."""
from __future__ import annotations

import pandas as pd


def _first_nonempty_col(df: pd.DataFrame, *cands: str, default: str | None = None) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return default


def compute_td_rate(weekly: pd.DataFrame, *, recent_window: int = 4) -> pd.DataFrame:
    """
    Robustly compute per-player TD features and recent usage (opportunities).

    Expects a weekly-level dataframe; handles varied column names from nfl_data_py.
    Returns columns:
      player, team, position (if available), games, two_plus, mean_td, recent_opps
    """
    df = weekly.copy()

    # -------- normalize key identifiers --------
    player_col = _first_nonempty_col(df, "player", "player_display_name", "full_name", "name")
    if not player_col:
        raise KeyError("Could not find a player name column in weekly data.")
    df["player"] = df[player_col].astype(str)

    team_col = _first_nonempty_col(df, "recent_team", "team", "posteam")
    df["team"] = df[team_col].astype(str) if team_col else ""

    pos_col = _first_nonempty_col(df, "position", "pos")
    df["position"] = df[pos_col].astype(str) if pos_col else ""

    week_col = _first_nonempty_col(df, "week", "game_week")
    if not week_col:
        # fallback: treat each row as its own "game index"
        df["week"] = range(1, len(df) + 1)
        week_col = "week"

    # -------- touchdowns (rushing + receiving) --------
    def _nz_int(colnames: list[str]) -> pd.Series:
        vals = None
        for c in colnames:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                vals = s if vals is None else (vals + s)
        return (vals if vals is not None else pd.Series(0, index=df.index)).astype(int)

    rushing_td = _nz_int(["rushing_td", "rush_td", "rushing_tds"])
    receiving_td = _nz_int(["receiving_td", "rec_td", "receiving_tds"])
    df["td_total"] = rushing_td + receiving_td

    # -------- "opportunities" = rush attempts + targets --------
    rush_att = _nz_int(["rushing_att", "rushing_attempts", "rush_att", "carries"])
    targets = _nz_int(["targets", "rec_targets"])
    df["opps"] = rush_att + targets

    # -------- aggregate per player --------
    agg = (
        df.groupby(["player", "team", "position"], as_index=False)
          .agg(
              games=(week_col, "nunique"),
              two_plus=("td_total", lambda s: (s >= 2).sum()),
              mean_td=("td_total", "mean"),
          )
    )

    # -------- recent usage (rolling last N games) --------
    # sort by week, compute per-player rolling mean of opportunities, take the last value
    df_sorted = df.sort_values([ "player", week_col ])
    df_sorted["recent_opps"] = (
        df_sorted.groupby("player")["opps"]
                 .transform(lambda s: s.rolling(min_periods=1, window=recent_window).mean())
    )
    last_recent = (
        df_sorted.groupby("player", as_index=False)["recent_opps"].last()
                 .rename(columns={"recent_opps": "recent_opps"})
    )

    out = agg.merge(last_recent, on="player", how="left")
    out["games"] = pd.to_numeric(out["games"], errors="coerce").fillna(1).astype(int)
    out["mean_td"] = pd.to_numeric(out["mean_td"], errors="coerce").fillna(0.0)
    out["recent_opps"] = pd.to_numeric(out["recent_opps"], errors="coerce").fillna(0.0)

    return out[["player", "team", "position", "games", "two_plus", "mean_td", "recent_opps"]]


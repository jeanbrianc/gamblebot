"""Feature engineering utilities for touchdown and sack props."""
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


def _nz_float(df: pd.DataFrame, colnames: list[str]) -> pd.Series:
    vals = None
    for c in colnames:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            vals = s if vals is None else (vals + s)
    return (vals if vals is not None else pd.Series(0.0, index=df.index)).astype(float)


def _team_dropback_sack_rates(season: int) -> pd.DataFrame:
    """Compute last season's team dropbacks and sack rate allowed with shrinkage."""
    from . import data  # local import to avoid circular

    try:
        pbp = data.load_pbp(season)
    except Exception:
        return pd.DataFrame(columns=["team", "dropbacks_per_game", "sack_rate"])

    dropbacks = _nz_float(pbp, ["dropback"])
    sacks = _nz_float(pbp, ["sack"])
    team_col = _first_nonempty_col(pbp, "posteam", "offense", "pos_team")
    week_col = _first_nonempty_col(pbp, "week", "game_week")
    if team_col is None or week_col is None:
        return pd.DataFrame(columns=["team", "dropbacks_per_game", "sack_rate"])

    agg = (
        pbp.assign(dropbacks=dropbacks, sacks=sacks)
        .groupby([team_col, week_col], as_index=False)
        .agg(dropbacks=("dropbacks", "sum"), sacks=("sacks", "sum"))
    )
    team_agg = (
        agg.groupby(team_col, as_index=False)
        .agg(dropbacks=("dropbacks", "sum"), sacks=("sacks", "sum"), games=(week_col, "nunique"))
    )
    team_agg["dropbacks_per_game"] = team_agg["dropbacks"] / team_agg["games"].clip(lower=1)
    team_agg["sack_rate"] = team_agg.apply(
        lambda r: (r["sacks"] / r["dropbacks"]) if r["dropbacks"] > 0 else 0.0, axis=1
    )

    league_dropbacks = team_agg["dropbacks_per_game"].mean()
    league_sack_rate = team_agg["sack_rate"].mean()
    prior_strength = 8.0
    w = team_agg["games"] / (team_agg["games"] + prior_strength)
    team_agg["dropbacks_per_game"] = w * team_agg["dropbacks_per_game"] + (1 - w) * league_dropbacks
    team_agg["sack_rate"] = w * team_agg["sack_rate"] + (1 - w) * league_sack_rate

    return team_agg.rename(columns={team_col: "team"})[["team", "dropbacks_per_game", "sack_rate"]]


def compute_sack_features(weekly: pd.DataFrame, season: int, *, recent_window: int = 4) -> pd.DataFrame:
    """Compute defender pass-rush features for the sacks report."""
    df = weekly.copy()

    player_col = _first_nonempty_col(df, "player", "player_display_name", "full_name", "name")
    if not player_col:
        raise KeyError("Could not find a player name column in weekly data.")
    df["player"] = df[player_col].astype(str)

    team_col = _first_nonempty_col(df, "recent_team", "team", "posteam")
    df["team"] = df[team_col].astype(str) if team_col else ""

    opp_col = _first_nonempty_col(df, "opponent", "opp", "defteam", "opp_team")
    df["opponent"] = df[opp_col].astype(str) if opp_col else ""

    pos_col = _first_nonempty_col(df, "position", "pos")
    df["position"] = df[pos_col].astype(str) if pos_col else ""

    week_col = _first_nonempty_col(df, "week", "game_week")
    if not week_col:
        df["week"] = range(1, len(df) + 1)
        week_col = "week"

    df["pressures"] = _nz_float(df, ["pressures", "qb_hits", "def_pressures"])
    df["prwin"] = _nz_float(df, ["prwin", "pass_rush_win_rate", "prwin_pct"]).clip(lower=0.0, upper=1.0)
    df["pass_rush_snaps"] = _nz_float(df, ["pass_rushes", "pass_rush_snaps", "prsnaps"])

    agg = (
        df.groupby(["player", "team", "opponent", "position"], as_index=False)
        .agg(
            pressures=("pressures", "sum"),
            prwin=("prwin", "mean"),
            pass_rush_snaps=("pass_rush_snaps", "sum"),
            games=(week_col, "nunique"),
        )
    )

    agg["pressures_per_game"] = agg["pressures"] / agg["games"].clip(lower=1)

    team_rates = _team_dropback_sack_rates(season - 1)
    agg = agg.merge(team_rates, left_on="opponent", right_on="team", how="left", suffixes=("", "_opp"))

    league_dropbacks = team_rates["dropbacks_per_game"].mean() if not team_rates.empty else 35.0
    league_sack_rate = team_rates["sack_rate"].mean() if not team_rates.empty else 0.07

    agg["opp_dropbacks"] = agg["dropbacks_per_game"].fillna(league_dropbacks)
    agg["opp_sack_rate_allowed"] = agg["sack_rate"].fillna(league_sack_rate)

    return agg[
        [
            "player",
            "team",
            "opponent",
            "position",
            "pressures_per_game",
            "prwin",
            "pass_rush_snaps",
            "opp_dropbacks",
            "opp_sack_rate_allowed",
        ]
    ]


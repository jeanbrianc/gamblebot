"""Data loading utilities using nfl_data_py with simple caching."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional
from urllib.error import HTTPError
import warnings

import pandas as pd
import requests_cache
from nfl_data_py import import_pbp_data, import_weekly_data

# Cache HTTP requests for 12 hours
requests_cache.install_cache("nfl_cache", expire_after=12 * 60 * 60)


@lru_cache(maxsize=32)
def load_pbp(season: int) -> pd.DataFrame:
    """Load play-by-play data for a season."""
    return import_pbp_data([season])


@lru_cache(maxsize=128)
def load_weekly_player_stats(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Load weekly player stats for a season and optionally a specific week.

    Strategy:
      1) Try nflverse weekly parquet for the requested season.
      2) If it's not published yet (HTTP 404), fall back to the most recent prior
         season's weekly data.
      3) If that also fails, attempt to construct minimal stats from play-by-play.
    """
    # -- 1) Primary path: requested season weekly parquet --
    try:
        df = import_weekly_data(years=[season])  # some versions don't accept weeks=
        if week is not None and "week" in df.columns:
            df = df[df["week"] == week]
        return _normalize_weekly_columns(df).reset_index(drop=True)
    except HTTPError as e:
        if getattr(e, "code", None) != 404:
            # A non-404 network error; re-raise
            raise
        # Weekly parquet for this season not found – continue to fallback
    except Exception:
        # Unexpected error path – continue to fallback
        pass

    # -- 2) Fallback: use the most recent prior season with weekly data --
    prior = _load_prior_season_weekly_any(season)
    if prior is not None:
        prior_season = int(prior["season"].iloc[0]) if "season" in prior.columns and not prior.empty else season - 1
        warnings.warn(
            f"Weekly data for season {season} not available yet; using season {prior_season} weekly dataset for features.",
            RuntimeWarning,
        )
        if week is not None and "week" in prior.columns:
            prior = prior[prior["week"] == week]
        return _normalize_weekly_columns(prior).reset_index(drop=True)

    # -- 3) Last resort: build minimal stats from PBP for the requested season/week --
    try:
        built = _weekly_player_stats_from_pbp(season=season, week=week)
        return _normalize_weekly_columns(built).reset_index(drop=True)
    except Exception as e:
        warnings.warn(
            f"Could not build fallback weekly stats from PBP for season {season}: {e}",
            RuntimeWarning,
        )
        # Return an empty frame with expected columns so downstream code doesn't crash.
        cols = ["player", "team", "week", "rush_att", "targets", "rushing_td", "receiving_td",
                "player_id", "player_display_name", "recent_team"]
        return pd.DataFrame(columns=cols)


def _load_prior_season_weekly_any(season: int, max_back: int = 5) -> Optional[pd.DataFrame]:
    """
    Try to load weekly data from prior seasons, up to `max_back` years back.
    Returns the first successfully loaded DataFrame, or None if none found.
    """
    for y in range(season - 1, season - 1 - max_back, -1):
        try:
            df = import_weekly_data(years=[y])
            if not df.empty:
                return df
        except HTTPError as e:
            if getattr(e, "code", None) == 404:
                continue
            raise
        except Exception:
            continue
    return None


def _weekly_player_stats_from_pbp(season: int, week: Optional[int]) -> pd.DataFrame:
    """
    Construct a minimal weekly player-stats table from play-by-play:
      - rush_att (carries), rushing_td
      - targets, receiving_td
    Output columns: player, team, week, rush_att, targets, rushing_td, receiving_td
    """
    pbp = import_pbp_data([season])
    if "week" in pbp.columns and week is not None:
        pbp = pbp[pbp["week"] == week]

    # Be robust to schema differences across seasons
    for col, alt in [
        ("rush_attempt", "rush"),
        ("rush_touchdown", "rush_td"),
        ("pass_attempt", "pass"),
        ("pass_touchdown", "pass_td"),
    ]:
        if col not in pbp.columns and alt in pbp.columns:
            pbp[col] = pbp[alt]

    # Ensure presence / types
    for col in ["rush_attempt", "rush_touchdown", "pass_attempt", "pass_touchdown"]:
        if col not in pbp.columns:
            pbp[col] = 0
        pbp[col] = pbp[col].fillna(0).astype(int)

    # Column names can vary slightly
    rush_name = "rusher_player_name" if "rusher_player_name" in pbp.columns else ("rusher" if "rusher" in pbp.columns else None)
    rec_name = "receiver_player_name" if "receiver_player_name" in pbp.columns else ("receiver" if "receiver" in pbp.columns else None)
    team_col = "posteam" if "posteam" in pbp.columns else ("pos_team" if "pos_team" in pbp.columns else ("offense" if "offense" in pbp.columns else None))

    # --- Rushing aggregation ---
    rush = pd.DataFrame(columns=["player", "team", "week", "rush_att", "rushing_td"])
    if rush_name and team_col:
        rush_mask = (pbp["rush_attempt"] == 1) & pbp[rush_name].notna()
        rush = (
            pbp.loc[rush_mask, [rush_name, team_col, "week", "rush_attempt", "rush_touchdown"]]
            .groupby([rush_name, team_col, "week"], as_index=False)
            .agg(rush_att=("rush_attempt", "sum"), rushing_td=("rush_touchdown", "sum"))
            .rename(columns={rush_name: "player", team_col: "team"})
        )

    # --- Receiving aggregation (targets & TDs) ---
    rec = pd.DataFrame(columns=["player", "team", "week", "targets", "receiving_td"])
    if rec_name and team_col:
        targ_mask = (pbp["pass_attempt"] == 1) & pbp[rec_name].notna()
        rec = (
            pbp.loc[targ_mask, [rec_name, team_col, "week", "pass_touchdown"]]
            .assign(targets=1)
            .groupby([rec_name, team_col, "week"], as_index=False)
            .agg(targets=("targets", "sum"), receiving_td=("pass_touchdown", "sum"))
            .rename(columns={rec_name: "player", team_col: "team"})
        )

    # --- Combine rushing & receiving into one player-week table ---
    out = pd.merge(rush, rec, on=["player", "team", "week"], how="outer")
    for col in ["rush_att", "rushing_td", "targets", "receiving_td"]:
        if col not in out.columns:
            out[col] = 0
    out[["rush_att", "rushing_td", "targets", "receiving_td"]] = (
        out[["rush_att", "rushing_td", "targets", "receiving_td"]].fillna(0).astype(int)
    )

    if week is not None and "week" in out.columns:
        out = out[out["week"] == week]
    return out.reset_index(drop=True)


def _normalize_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to what downstream code expects and keep aliases:
      Required downstream (features): player_id, player_display_name, recent_team
      Also provide: player, team, week, rush_att, targets, rushing_td, receiving_td
    """
    df = df.copy()

    # --- Player name ---
    player_candidates = [
        "player_display_name", "player_name", "full_name", "player", "name"
    ]
    player_col = next((c for c in player_candidates if c in df.columns), None)
    if player_col is None:
        df["player"] = ""
        df["player_display_name"] = ""
    else:
        # Ensure 'player' exists and keep original display name if present
        df["player"] = df[player_col].astype("string")
        if "player_display_name" not in df.columns:
            df["player_display_name"] = df["player"]

    # --- Team columns ---
    team_candidates = ["recent_team", "team", "player_team", "posteam"]
    team_col = next((c for c in team_candidates if c in df.columns), None)
    if team_col is None:
        df["team"] = pd.NA
        df["recent_team"] = pd.NA
    else:
        df["team"] = df[team_col]
        if "recent_team" not in df.columns:
            df["recent_team"] = df["team"]

    # --- Week column ---
    if "week" not in df.columns and "game_week" in df.columns:
        df["week"] = df["game_week"]
    if "week" not in df.columns:
        df["week"] = 0
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

    # --- Touchdowns and usage stats ---
    # rushing
    if "rushing_td" not in df.columns:
        for c in ["rushing_tds", "rush_td"]:
            if c in df.columns:
                df["rushing_td"] = df[c]
                break
    if "rushing_td" not in df.columns:
        df["rushing_td"] = 0

    # receiving
    if "receiving_td" not in df.columns:
        for c in ["receiving_tds", "rec_td"]:
            if c in df.columns:
                df["receiving_td"] = df[c]
                break
    if "receiving_td" not in df.columns:
        df["receiving_td"] = 0

    # rush attempts
    if "rush_att" not in df.columns:
        for c in ["rushing_attempts", "carries"]:
            if c in df.columns:
                df["rush_att"] = df[c]
                break
    if "rush_att" not in df.columns:
        df["rush_att"] = 0

    # targets
    if "targets" not in df.columns:
        df["targets"] = 0

    for col in ["rushing_td", "receiving_td", "rush_att", "targets"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- Player ID ---
    if "player_id" not in df.columns:
        # Synthetic but stable-ish id for grouping if missing
        df["player_id"] = (df["player"].fillna("").astype(str) + ":" + df["team"].fillna("").astype(str))

    # Ensure string types where useful
    df["player_display_name"] = df["player_display_name"].astype("string")
    df["player"] = df["player"].astype("string")

    return df


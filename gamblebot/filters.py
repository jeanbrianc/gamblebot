"""Filtering helpers for starters and injuries."""
from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd

# Injury statuses we exclude if we find them
_EXCLUDE_STATUSES = {
    "out", "doubtful", "inactive", "questionable - inactive",
    "ir", "injured reserve", "pup", "physically unable to perform",
    "nfi", "non-football injury", "suspended",
}

_NAME_CLEAN_RE = re.compile(r"[^a-z]+")


def _clean_name(x: str) -> str:
    return _NAME_CLEAN_RE.sub("", str(x).lower())


def load_injury_status(season: int, week: int) -> pd.DataFrame:
    """
    Best-effort pull of weekly injury statuses from nfl_data_py.
    Returns columns: player (str), injury_status (str).
    If the source is unavailable, returns empty df.
    """
    try:
        from nfl_data_py import import_injury_reports  # newer versions
        inj = import_injury_reports([season])
    except Exception:
        try:
            from nfl_data_py import import_injuries  # older alias
            inj = import_injuries([season])
        except Exception:
            return pd.DataFrame(columns=["player", "injury_status"])

    # normalize columns across versions
    df = pd.DataFrame(inj)
    if df.empty:
        return pd.DataFrame(columns=["player", "injury_status"])

    # typical columns: "week", "player_name", "gsis_status" / "report_status" / "injury_status"
    name_col = "player_name" if "player_name" in df.columns else ("player" if "player" in df.columns else None)
    if not name_col:
        return pd.DataFrame(columns=["player", "injury_status"])

    status_col = None
    for c in ("gsis_status", "report_status", "injury_status", "status"):
        if c in df.columns:
            status_col = c
            break
    if not status_col:
        return pd.DataFrame(columns=["player", "injury_status"])

    # filter to the target week if present
    if "week" in df.columns:
        df = df[df["week"] == int(week)]

    out = pd.DataFrame({
        "player": df[name_col].astype(str),
        "injury_status": df[status_col].astype(str).str.lower().str.strip(),
    }).dropna()

    # keep most pessimistic status if multiple entries
    order = {s: i for i, s in enumerate([
        "suspended", "ir", "injured reserve", "pup", "physically unable to perform",
        "nfi", "non-football injury", "out", "inactive", "questionable - inactive",
        "doubtful", "questionable", "probable", "cleared", "healthy", "",
    ])}
    out["rank"] = out["injury_status"].map(lambda s: order.get(s, len(order)))
    out = out.sort_values("rank").drop_duplicates(subset=["player"], keep="first").drop(columns=["rank"])

    return out.reset_index(drop=True)


def apply_filters(
    model_df: pd.DataFrame,
    *,
    positions: Optional[Iterable[str]] = None,
    min_recent_opps: float = 3.0,
    exclude_injured: bool = True,
    season: Optional[int] = None,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filters:
      - positions: keep only these positions (case-insensitive), e.g. {"RB","WR","TE"}.
      - min_recent_opps: keep players with recent avg opportunities >= threshold.
      - exclude_injured: if True and season/week provided, drop players with 'bad' statuses.
    """
    df = model_df.copy()

    # Position filter (if model_df has 'position')
    if positions and "position" in df.columns:
        keep = {p.strip().upper() for p in positions if p.strip()}
        if keep:
            df = df[df["position"].astype(str).str.upper().isin(keep)]

    # Usage filter
    if "recent_opps" in df.columns:
        df = df[pd.to_numeric(df["recent_opps"], errors="coerce").fillna(0) >= float(min_recent_opps)]

    # Injury filter
    if exclude_injured and (season is not None) and (week is not None):
        inj = load_injury_status(season, week)
        if not inj.empty:
            inj["key"] = inj["player"].map(_clean_name)
            df["key"] = df["player"].map(_clean_name)
            merged = df.merge(inj.rename(columns={"injury_status": "injury_status_raw"}),
                              on="key", how="left")
            merged["injury_status"] = merged["injury_status_raw"].fillna("")
            mask_bad = merged["injury_status"].isin(_EXCLUDE_STATUSES)
            df = merged.loc[~mask_bad, model_df.columns]  # back to original cols

    return df.reset_index(drop=True)


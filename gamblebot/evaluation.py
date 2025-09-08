"""Utilities for logging predictions and evaluating outcomes."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import data

DEFAULT_LOG = Path("predictions.csv")


def record_predictions(df: pd.DataFrame, season: int, week: int, path: Path = DEFAULT_LOG) -> None:
    """Append predictions to a CSV log with season/week metadata."""
    log_df = df.copy()
    log_df["season"] = season
    log_df["week"] = week
    log_df["timestamp"] = pd.Timestamp.utcnow()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    log_df.to_csv(path, mode="a", header=header, index=False)


def evaluate_predictions(season: int, week: int, path: Path = DEFAULT_LOG) -> Tuple[pd.DataFrame, dict]:
    """Evaluate logged predictions against actual 2+ TD outcomes."""
    if not path.exists():
        return pd.DataFrame(), {}
    preds = pd.read_csv(path)
    preds = preds[(preds["season"] == season) & (preds["week"] == week)]
    if preds.empty:
        return preds, {}
    weekly = data.load_weekly_player_stats(season, week=week)
    weekly = weekly.assign(
        tds=pd.to_numeric(weekly.get("rushing_td", 0), errors="coerce").fillna(0)
        + pd.to_numeric(weekly.get("receiving_td", 0), errors="coerce").fillna(0)
    )[["player", "tds"]]
    merged = preds.merge(weekly, on="player", how="left")
    merged["tds"] = merged["tds"].fillna(0).astype(int)
    merged["hit"] = merged["tds"] >= 2
    merged["profit"] = merged.apply(
        lambda r: r["stake_units"] * (r["odds"] - 1) if r["hit"] else -r["stake_units"],
        axis=1,
    )
    merged["brier"] = (merged["model_prob"] - merged["hit"].astype(float)) ** 2
    metrics = {
        "n": int(len(merged)),
        "hit_rate": float(merged["hit"].mean()),
        "roi": float(merged["profit"].sum() / merged["stake_units"].sum()) if merged["stake_units"].sum() else 0.0,
        "brier": float(merged["brier"].mean()),
    }
    return merged, metrics

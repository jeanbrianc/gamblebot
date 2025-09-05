"""Odds fetching utilities for sportsbook integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
import requests_cache
import requests


@dataclass
class TheOddsAPIClient:
    """Simple client for TheOddsAPI."""

    api_key: str
    base_url: str = "https://api.the-odds-api.com/v4"

    def __post_init__(self) -> None:
        self.session = requests_cache.CachedSession("odds_cache", expire_after=5 * 60)

    def fetch_two_td_odds(
        self, season: int, week: int, books: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """Fetch 2+ TD odds for all players for a given week."""
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "player_two_td",
            "oddsFormat": "american",
        }
        if books:
            params["bookmakers"] = ",".join(books)
        url = f"{self.base_url}/sports/americanfootball_nfl/odds"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for event in data:
            for bm in event.get("bookmakers", []):
                book = bm["key"]
                for market in bm.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        player = outcome.get("name")
                        price = outcome.get("price")
                        rows.append({
                            "player": player,
                            "book": book,
                            "odds": price,
                        })
        df = pd.DataFrame(rows)
        df["implied_prob"] = df["odds"].apply(american_to_implied)
        return df


def american_to_implied(odds: int | float) -> float:
    """Convert American odds to implied probability."""
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 - 100 / odds


def normalize_book_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best (lowest implied probability) odds per player."""
    idx = df.groupby("player")["implied_prob"].idxmin()
    return df.loc[idx].reset_index(drop=True)

"""Odds fetching + normalization for 2+ TD props via The Odds API (v4)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
import requests


API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
REGIONS = "us,us2"  # wider US coverage
ODDS_FORMAT = "american"


# ---------- odds utils ----------

def american_to_decimal(american: int | float) -> float:
    a = float(american)
    if a >= 100:
        return 1.0 + (a / 100.0)
    if a <= -100:
        return 1.0 + (100.0 / abs(a))
    # fall back (shouldn't happen for real prices)
    return 1.0


def american_to_implied(american: int | float) -> float:
    a = float(american)
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return abs(a) / (abs(a) + 100.0)


def _first_thursday_of_september(season: int) -> datetime:
    d = datetime(season, 9, 1, tzinfo=timezone.utc)
    # weekday: Mon=0 ... Sun=6; Thu=3
    offset = (3 - d.weekday()) % 7
    return d + timedelta(days=offset)


def _nfl_week_window_utc(season: int, week: int, widen_days: int = 2) -> tuple[str, str]:
    # Thursday of kickoff week, then add (week-1)*7 days
    kickoff_thu = _first_thursday_of_september(season)
    start = kickoff_thu + timedelta(days=7 * (week - 1))
    # widen a bit to be lenient with posting schedules
    start = (start - timedelta(days=widen_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7 + 2 * widen_days)
    # return ISO8601 Z times
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


# ---------- player-name extraction ----------

_OVER_UNDER = re.compile(r"\b(Over|Under)\b", re.I)

def _extract_player_name(outcome: dict) -> Optional[str]:
    """
    Try multiple fields; sanitize cases where name=='Over' and player lives in 'description'.
    Handles:
      - "Patrick Mahomes Over 1.5"
      - "Over 1.5 - Patrick Mahomes"
      - "Over 1.5 (Patrick Mahomes)"
      - raw participant/player/label
    """
    candidates = [
        outcome.get("participant"),
        outcome.get("player"),
        outcome.get("description"),
        outcome.get("label"),
        outcome.get("name"),
    ]
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue

        # If it's literally "Over"/"Under", skip to other fields
        if text.lower() in ("over", "under"):
            continue

        # Pattern 1: "<Player> Over/Under ..."
        m = re.match(r"^(?P<player>.+?)\s+(Over|Under)\b.*$", text, flags=re.I)
        if m:
            return m.group("player").strip()

        # Pattern 2: "Over/Under ... - <Player>"
        m = re.match(r"^(?:Over|Under)\b.*?[-–]\s*(?P<player>.+)$", text, flags=re.I)
        if m:
            return m.group("player").strip()

        # Pattern 3: "Over/Under ... (<Player>)"
        m = re.match(r"^(?:Over|Under)\b.*?\((?P<player>.+)\)$", text, flags=re.I)
        if m:
            return m.group("player").strip()

        # Pattern 4: "Over 1.5 <Player>"
        m = re.match(r"^(?:Over|Under)\b.*?\s(?P<player>[A-Za-z][A-Za-z .'\-]+)$", text, flags=re.I)
        if m:
            return m.group("player").strip()

        # Otherwise, treat the text as the player name
        return text

    return None


# ---------- client ----------

class TheOddsAPIClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key
        self.http = session or requests.Session()

    def _params(self, **extra: object) -> dict[str, object]:
        p: dict[str, object] = {
            "apiKey": self.api_key,
            "regions": REGIONS,
            "oddsFormat": ODDS_FORMAT,
        }
        p.update(extra)
        return p

    def _events_for_week(self, season: int, week: int) -> list[dict]:
        ts_from, ts_to = _nfl_week_window_utc(season, week, widen_days=2)
        resp = self.http.get(
            f"{API_BASE}/sports/{SPORT}/events",
            params=self._params(commenceTimeFrom=ts_from, commenceTimeTo=ts_to),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def fetch_two_td_odds(
        self,
        season: int,
        week: int,
        books: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        """
        Returns a list of rows:
        {event_id, home_team, away_team, book, player, odds, implied_prob}
        """
        events = self._events_for_week(season, week)
        books_set = set(b.lower() for b in books) if books else None

        rows: list[dict] = []
        # Primary market = X+ touchdowns (“Over only”), want >= 1.5 for 2+
        # Fallback = alternate rush+rec TDs where point >= 2.0
        market_order = ("player_tds_over", "player_rush_reception_tds_alternate")

        for ev in events:
            got_any_for_event = False
            for market in market_order:
                try:
                    r = self.http.get(
                        f"{API_BASE}/sports/{SPORT}/events/{ev['id']}/odds",
                        params=self._params(markets=market),
                        timeout=25,
                    )
                    if r.status_code in (404, 422):
                        continue
                    r.raise_for_status()
                    data = r.json()
                except requests.HTTPError:
                    continue

                for bm in data.get("bookmakers", []):
                    bm_key = bm.get("key", "").lower()
                    if books_set and bm_key not in books_set:
                        continue
                    for mk in bm.get("markets", []):
                        if mk.get("key") != market:
                            continue
                        for out in mk.get("outcomes", []):
                            price = out.get("price")
                            point = out.get("point")
                            if price is None:
                                continue

                            # Ensure it's 2+ touchdowns
                            if market == "player_tds_over":
                                if point is None or float(point) < 1.5:
                                    continue
                            else:  # player_rush_reception_tds_alternate
                                if point is None or float(point) < 2.0:
                                    continue

                            player = _extract_player_name(out)
                            if not player:
                                continue

                            rows.append(
                                {
                                    "event_id": ev["id"],
                                    "home_team": ev.get("home_team"),
                                    "away_team": ev.get("away_team"),
                                    "book": bm_key,
                                    "player": player,
                                    "odds": int(price),
                                    "implied_prob": american_to_implied(price),
                                }
                            )
                            got_any_for_event = True
                if got_any_for_event:
                    break

        return rows


def normalize_book_odds(rows: list[dict]) -> pd.DataFrame:
    """
    Convert the raw list from fetch_two_td_odds into a de-duped frame
    with the columns expected by the rest of the pipeline.
    Picks the *best* (longest) price per player across books.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep longest price (lowest implied prob) per player
    df = df.sort_values("implied_prob").drop_duplicates(subset=["player"], keep="first")
    return df[["player", "odds", "implied_prob"]].reset_index(drop=True)


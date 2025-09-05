"""Odds fetching + normalization for 2+ TD props via The Odds API (v4)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
import requests


API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
REGIONS = "us,us2"  # widen US coverage
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

def _extract_player_name(outcome: dict) -> Optional[str]:
    """
    Handle books that put player in 'participant'/'player'/'description'/'label'
    and leave name as 'Over'/'Under'. Also parse strings like:
      - 'Patrick Mahomes Over 1.5'
      - 'Over 1.5 - Patrick Mahomes'
      - 'Over 1.5 (Patrick Mahomes)'
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
        if text.lower() in ("over", "under"):
            continue

        # <Player> Over/Under ...
        m = re.match(r"^(?P<player>.+?)\s+(Over|Under)\b.*$", text, flags=re.I)
        if m:
            return m.group("player").strip()
        # Over/Under ... - <Player>
        m = re.match(r"^(?:Over|Under)\b.*?[-–]\s*(?P<player>.+)$", text, flags=re.I)
        if m:
            return m.group("player").strip()
        # Over/Under ... (<Player>)
        m = re.match(r"^(?:Over|Under)\b.*?\((?P<player>.+)\)$", text, flags=re.I)
        if m:
            return m.group("player").strip()
        # Over 1.5 <Player>
        m = re.match(r"^(?:Over|Under)\b.*?\s(?P<player>[A-Za-z][A-Za-z .'\-]+)$", text, flags=re.I)
        if m:
            return m.group("player").strip()

        # Otherwise assume this string is the player
        return text
    return None


def _is_two_plus(market_key: str, outcome: dict) -> tuple[bool, Optional[float]]:
    """
    Return (is_two_plus, line_point) where line_point is the numeric threshold if present.
    Rules:
      - player_tds_over: keep only 1.5 ≤ point < 2.5 (Over 1.5)
      - player_rush_reception_tds_alternate: keep only point ≈ 2.0 (±0.01) or text says '2+'
    """
    pt = outcome.get("point")
    desc_fields = [outcome.get("description"), outcome.get("label"), outcome.get("name")]
    desc = " ".join([s for s in desc_fields if isinstance(s, str)]).lower()

    try:
        pval = float(pt) if pt is not None else None
    except Exception:
        pval = None

    if market_key == "player_tds_over":
        if pval is not None:
            return (1.5 <= pval < 2.5, pval)
        # fallback text parse
        return (("1.5" in desc) or ("2+" in desc) or ("two or more" in desc)), pval

    # alternate TD totals (rush+rec)
    if pval is not None:
        return (abs(pval - 2.0) < 0.01, pval)
    return (("2+" in desc) or ("2 or more" in desc) or ("two or more" in desc)), pval


def _is_one_sack(market_key: str, outcome: dict) -> tuple[bool, Optional[float]]:
    """Return (is_one_plus, line_point) for 1+ sack markets."""
    pt = outcome.get("point")
    desc_fields = [outcome.get("description"), outcome.get("label"), outcome.get("name")]
    desc = " ".join([s for s in desc_fields if isinstance(s, str)]).lower()

    try:
        pval = float(pt) if pt is not None else None
    except Exception:
        pval = None

    if pval is not None:
        return (0.45 <= pval <= 1.05, pval)
    return ("1+" in desc or "1 or more" in desc or "one or more" in desc, pval)


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
        Returns rows for **exactly 2+ TD**:
        {event_id, home_team, away_team, book, player, odds, implied_prob, line, market}
        """
        events = self._events_for_week(season, week)
        books_set = set(b.lower() for b in books) if books else None

        rows: list[dict] = []
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
                            if price is None:
                                continue

                            is_two_plus, line_point = _is_two_plus(market, out)
                            if not is_two_plus:
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
                                    "line": float(line_point) if line_point is not None else None,
                                    "market": market,
                                }
                            )
                            got_any_for_event = True
                if got_any_for_event:
                    break

        return rows

    def fetch_sack_odds(
        self,
        season: int,
        week: int,
        books: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        """Return rows for **1+ sack** props."""
        events = self._events_for_week(season, week)
        books_set = set(b.lower() for b in books) if books else None

        rows: list[dict] = []
        market_order = ("player_sacks_over", "player_sacks")

        for ev in events:
            got_any = False
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
                        if "sack" not in str(mk.get("key", "")):
                            continue
                        for out in mk.get("outcomes", []):
                            price = out.get("price")
                            if price is None:
                                continue
                            is_one, line_point = _is_one_sack(market, out)
                            if not is_one:
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
                                    "line": float(line_point) if line_point is not None else None,
                                    "market": market,
                                }
                            )
                            got_any = True
                if got_any:
                    break

        return rows


def normalize_book_odds(rows: list[dict]) -> pd.DataFrame:
    """Deduplicate to one best (longest) price per player for the **2+ TD** line."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("implied_prob").drop_duplicates(subset=["player"], keep="first")
    return df[["player", "odds", "implied_prob"]].reset_index(drop=True)


def normalize_sack_odds(rows: list[dict]) -> pd.DataFrame:
    """Deduplicate to best price per player for the **1+ sack** line."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("implied_prob").drop_duplicates(subset=["player"], keep="first")
    return df[["player", "line", "odds", "implied_prob"]].reset_index(drop=True)


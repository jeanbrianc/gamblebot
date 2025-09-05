"""Data loading utilities using nfl_data_py with simple caching."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

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
    """Load weekly player stats for a season and optionally a week."""
    weeks = [week] if week is not None else None
    return import_weekly_data(years=[season], weeks=weeks)

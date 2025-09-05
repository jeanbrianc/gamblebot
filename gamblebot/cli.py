"""Command line interface for two touchdown report."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import click

from . import data, features, model, odds, reporting, staking


@click.group()
def main() -> None:
    """Gamblebot CLI."""


@main.command()
@click.option("--season", type=int, required=True, help="Season year e.g. 2025")
@click.option("--week", type=int, required=True, help="Week number")
@click.option("--top", type=int, default=10, show_default=True, help="Top N players")
@click.option("--books", type=str, default="", help="Comma separated list of books")
@click.option("--kelly-fraction", type=float, default=0.5, show_default=True)
@click.option("--unit-size", type=float, default=1.0, show_default=True)
@click.option("--csv", type=click.Path(path_type=Path), help="Export to CSV")
@click.option("--html", type=click.Path(path_type=Path), help="Export to HTML")
@click.option("--png", type=click.Path(path_type=Path), help="Export to PNG")
def report(
    season: int,
    week: int,
    top: int,
    books: str,
    kelly_fraction: float,
    unit_size: float,
    csv: Optional[Path],
    html: Optional[Path],
    png: Optional[Path],
) -> None:
    """Generate a top-N report for 2+ TD scorer edges."""
    api_key = os.environ.get("THEODDS_API_KEY")
    if not api_key:
        raise click.UsageError("THEODDS_API_KEY environment variable not set")

    weekly = data.load_weekly_player_stats(season)
    td_rate = features.compute_td_rate(weekly)
    model_df = model.add_model_probability(td_rate)

    client = odds.TheOddsAPIClient(api_key)
    books_list = [b.strip() for b in books.split(",") if b.strip()] or None
    odds_df = client.fetch_two_td_odds(season, week, books_list)
    odds_df = odds.normalize_book_odds(odds_df)

    merged = model_df.merge(odds_df, on="player")
    merged = staking.add_edge_and_stake(
        merged, kelly_fraction=kelly_fraction, unit_size=unit_size
    )
    merged = merged.sort_values("edge", ascending=False).head(top)

    reporting.display_report(merged)
    reporting.export_report(merged, csv=csv, html=html, png=png)


if __name__ == "__main__":  # pragma: no cover
    main()

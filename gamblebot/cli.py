"""Command line interface for two touchdown report."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from . import data, features, model, odds, reporting, staking
from .filters import apply_filters, apply_passrush_filters


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
@click.option(
    "--dump-odds",
    is_flag=True,
    help="Print all available 2+ TD player odds (by book) for the selected week and exit.",
)
@click.option(
    "--positions",
    type=str,
    default="RB,WR,TE",
    show_default=True,
    help="Comma-separated list of positions to include (e.g. RB,WR,TE or add QB).",
)
@click.option(
    "--min-recent-opps",
    type=float,
    default=3.0,
    show_default=True,
    help="Minimum recent average opportunities (rush attempts + targets) over the last few games.",
)
@click.option(
    "--no-exclude-injured",
    is_flag=True,
    help="If set, do NOT exclude players with out/doubtful/IR/etc statuses.",
)
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
    dump_odds: bool,
    positions: str,
    min_recent_opps: float,
    no_exclude_injured: bool,
) -> None:
    """Generate a top-N report for 2+ TD scorer edges, or dump raw odds with --dump-odds."""
    api_key = os.environ.get("THEODDS_API_KEY")
    if not api_key:
        raise click.UsageError("THEODDS_API_KEY environment variable not set")

    client = odds.TheOddsAPIClient(api_key)
    books_list = [b.strip() for b in books.split(",") if b.strip()] or None

    # Dump mode: print *all* 2+ TD player odds for this week and exit
    if dump_odds:
        rows = client.fetch_two_td_odds(season, week, books_list)
        df = pd.DataFrame(rows)
        if df.empty:
            click.echo("No 2+ TD player odds found for the selected week/filters.")
            return
        cols = [c for c in ["player", "book", "odds", "implied_prob", "line", "market", "event_id", "home_team", "away_team"] if c in df.columns]
        df = df[cols].sort_values(["player", "book"]).reset_index(drop=True)
        click.echo(df.to_string(index=False))
        return

    # Normal pipeline
    weekly = data.load_weekly_player_stats(season, week=week)
    feats = features.compute_td_rate(weekly)
    model_df = model.add_model_probability(feats)

    # Apply filters for starters & injuries
    pos_list = [p.strip() for p in positions.split(",") if p.strip()]
    filtered = apply_filters(
        model_df,
        positions=pos_list,
        min_recent_opps=min_recent_opps,
        exclude_injured=(not no_exclude_injured),
        season=season,
        week=week,
    )

    rows = client.fetch_two_td_odds(season, week, books_list)
    odds_df = odds.normalize_book_odds(rows)

    merged = filtered.merge(odds_df, on="player")
    merged = staking.add_edge_and_stake(
        merged, kelly_fraction=kelly_fraction, unit_size=unit_size
    )
    merged = merged.sort_values("edge", ascending=False).head(top)

    reporting.display_report(merged)
    reporting.export_report(merged, csv=csv, html=html, png=png)


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
@click.option("--dump-odds", is_flag=True, help="Print all available 1+ sack lines and exit")
@click.option(
    "--positions",
    type=str,
    default="EDGE,DL,LB",
    show_default=True,
    help="Comma-separated list of defensive positions to include.",
)
@click.option(
    "--min-passrush-snaps",
    type=float,
    default=20.0,
    show_default=True,
    help="Minimum recent pass-rush snaps.",
)
@click.option("--no-exclude-injured", is_flag=True, help="Do NOT exclude injured players")
def sacks(
    season: int,
    week: int,
    top: int,
    books: str,
    kelly_fraction: float,
    unit_size: float,
    csv: Optional[Path],
    html: Optional[Path],
    png: Optional[Path],
    dump_odds: bool,
    positions: str,
    min_passrush_snaps: float,
    no_exclude_injured: bool,
) -> None:
    """Generate a top-N report for defenders to record 1+ sack."""
    api_key = os.environ.get("THEODDS_API_KEY")
    if not api_key:
        raise click.UsageError("THEODDS_API_KEY environment variable not set")

    client = odds.TheOddsAPIClient(api_key)
    books_list = [b.strip() for b in books.split(",") if b.strip()] or None

    if dump_odds:
        rows = client.fetch_sack_odds(season, week, books_list)
        df = pd.DataFrame(rows)
        if df.empty:
            click.echo("No 1+ sack odds found for the selected week/filters.")
            return
        cols = [
            c
            for c in [
                "player",
                "book",
                "odds",
                "implied_prob",
                "line",
                "market",
                "event_id",
                "home_team",
                "away_team",
            ]
            if c in df.columns
        ]
        df = df[cols].sort_values(["player", "book"]).reset_index(drop=True)
        click.echo(df.to_string(index=False))
        return

    weekly = data.load_weekly_player_stats(season, week=week)
    feats = features.compute_sack_features(weekly, season)
    model_df = model.add_sack_model_probability(feats)

    pos_list = [p.strip() for p in positions.split(",") if p.strip()]
    filtered = apply_passrush_filters(
        model_df,
        positions=pos_list,
        min_passrush_snaps=min_passrush_snaps,
        exclude_injured=(not no_exclude_injured),
        season=season,
        week=week,
    )

    rows = client.fetch_sack_odds(season, week, books_list)
    odds_df = odds.normalize_sack_odds(rows)

    merged = filtered.merge(odds_df, on="player")
    merged = staking.add_edge_and_stake(
        merged, kelly_fraction=kelly_fraction, unit_size=unit_size
    )
    merged = merged.sort_values("edge", ascending=False).head(top)

    reporting.display_report(merged)
    reporting.export_report(merged, csv=csv, html=html, png=png)


if __name__ == "__main__":  # pragma: no cover
    main()


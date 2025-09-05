"""Reporting utilities for console and file outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

try:  # optional dependency for PNG export
    import dataframe_image as dfi
except Exception:  # pragma: no cover - best effort
    dfi = None


console = Console()


def format_percentage(x: float) -> str:
    return f"{x:.1%}"


def display_report(df: pd.DataFrame) -> None:
    table = Table(show_header=True, header_style="bold")
    for col in ["team", "player", "model_prob", "odds", "implied_prob", "edge", "stake_units"]:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(
            row["team"],
            row["player"],
            format_percentage(row["model_prob"]),
            str(row["odds"]),
            format_percentage(row["implied_prob"]),
            format_percentage(row["edge"]),
            f"{row['stake_units']:.2f}",
        )
    console.print(table)


def export_report(df: pd.DataFrame, csv: Optional[Path] = None, html: Optional[Path] = None, png: Optional[Path] = None) -> None:
    if csv:
        df.to_csv(csv, index=False)
    if html:
        df.to_html(html, index=False)
    if png and dfi is not None:
        dfi.export(df, png)


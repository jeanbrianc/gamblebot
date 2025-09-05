# Gamblebot

Command line tool to identify NFL players with the best value on 2+ touchdown scorer props.

## Usage

Install dependencies and ensure an environment variable `THEODDS_API_KEY` is set for [TheOddsAPI](https://theoddsapi.com/).

```
poetry install  # or pip install .
export THEODDS_API_KEY=your_key
```

Run the weekly report:

```
two-td report --season 2025 --week 1 --top 10 --books fanduel,draftkings
```

The command prints a ranked table and optionally exports to CSV/HTML/PNG.

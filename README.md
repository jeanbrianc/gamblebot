# 2+ TD Report

Generate a ranked list of NFL players with value to score **2+ touchdowns** in a given week, combining a simple Poisson model with sportsbook prices.

---

## Prerequisites

Set your The Odds API key:

```bash
export THEODDS_API_KEY="your_api_key_here"
```

Run commands with Poetry from the project root:

```bash
# Value-focused edge report
poetry run gamblebot report --season <YEAR> --week <WEEK>

# Highest-probability TD scorers
poetry run gamblebot likely --season <YEAR> --week <WEEK>
```

After games are played you can evaluate logged predictions:

```
poetry run gamblebot evaluate --season <YEAR> --week <WEEK>
```

---

## Options

* `--season INT` **(required)**: Season year (e.g., `2025`).
* `--week INT` **(required)**: Week number (e.g., `1`).
* `--top INT` (default: `10`): Number of rows to display in the final report.
* `--books STR`: Comma-separated sportsbook keys to include (e.g., `fanduel,draftkings,caesars`).
  If omitted, all available US books are considered.
* `--kelly-fraction FLOAT` (default: `0.5`): Fractional Kelly sizing.
* `--unit-size FLOAT` (default: `1.0`): Multiplier for Kelly (your unit size).
* `--csv PATH`: Export final table to CSV.
* `--html PATH`: Export final table to HTML.
* `--png PATH`: Export final table to PNG.
* `--log PATH` (default: `predictions.csv`): Append predictions to this CSV for later evaluation.
* `--dump-odds` (flag): Print **all posted 2+ TD lines** (by book) for the chosen week and exit (great for sanity checks).
* `--positions STR` (default: `RB,WR,TE`): Comma-separated positions to include (add `QB` to include quarterbacks).
* `--min-recent-opps FLOAT` (default: `3.0`): Minimum recent average opportunities (rush attempts + targets) over the last few games; filters out low-usage players.
* `--no-exclude-injured` (flag): Disable the injury filter (by default, players listed as out/doubtful/IR/PUP/etc for that week are excluded).

The `likely` subcommand accepts the same options (except `--dump-odds`) but sorts by `model_prob` instead of `edge`.

### Evaluation

* `--season INT` **(required)**: Season year to evaluate.
* `--week INT` **(required)**: Week number to evaluate.
* `--log PATH` (default: `predictions.csv`): Prediction log file to read.

---

## What the tool does

* **Markets:** Pulls only true **2+ TD** prices (e.g., *Over 1.5 TDs* or *alt 2.0*), explicitly excluding 3+ TD lines.
* **Model:** Uses per-player weekly stats from the most recent available season to estimate the chance of **2+ TD** via a Poisson model with empirical-Bayes shrinkage.
* **Filters:** Keeps likely starters (via recent opportunities) and excludes injured players by default.
* **Merge:** Joins model probabilities with the **best** available price per player, computes implied probability, edge, and a Kelly stake.

> **Note:** If current-season weekly stats aren’t published yet, the tool falls back to **last season’s** weekly dataset for features (you’ll see a warning). Odds and injury info are always for the requested `--week`.

---

## Examples

Top 20 across all books:

```bash
poetry run gamblebot report --season 2025 --week 1 --top 20
```

Restrict to specific books and export:

```bash
poetry run gamblebot report --season 2025 --week 1 \
  --books fanduel,draftkings \
  --csv week1.csv --html week1.html
```

Stricter “starter” filter (≥5 recent opps) and include QBs:

```bash
poetry run gamblebot report --season 2025 --week 1 \
  --positions RB,WR,TE,QB --min-recent-opps 5
```

Show every posted **2+ TD** price (debug/sanity check):

```bash
poetry run gamblebot report --season 2025 --week 1 --dump-odds
```

List the most likely 2+ TD scorers (highest model probabilities):

```bash
poetry run gamblebot likely --season 2025 --week 1 --top 20
```

Evaluate prior predictions once results are known:

```
poetry run gamblebot evaluate --season 2025 --week 1
```

---

## Output Columns

* `team` – Player’s team.
* `player` – Player name.
* `model_prob` – Model probability of **2+ TD** this week.
* `odds` – Best American odds for **2+ TD**.
* `implied_prob` – Probability implied by the odds.
* `edge` – `model_prob - implied_prob`.
* `stake_units` – Kelly stake (scaled by `--kelly-fraction` and `--unit-size`).


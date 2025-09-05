# Gamblebot

Command line tool to identify NFL players with the best value on 2+ touchdown scorer props.

## Usage

Install the package into a Python 3.10+ environment and set an environment
variable `THEODDS_API_KEY` for
[TheOddsAPI](https://theoddsapi.com/). On macOS this can be done with the
system Python:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export THEODDS_API_KEY=your_key
```

Run the weekly report:

```
two-td report --season 2025 --week 1 --top 10 --books fanduel,draftkings
```

The command prints a ranked table and optionally exports to CSV/HTML/PNG.

You can also invoke the CLI without installing by using the module form:

```
python -m gamblebot report --season 2025 --week 1
```

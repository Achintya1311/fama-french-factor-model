"""Does Stock Stalker's screen survive factor adjustment? (Day 5)

Stock Stalker's screener ranks NSE candidates purely on price/volume
technicals (trend + momentum + volume). This module answers the question
Day 3's traps already flagged as the point of this whole repo: is that
ranking picking up a genuine edge, or just unpriced factor exposure (beta,
size, value tilt) that Ken French's own factors would explain away?

It reads Stock Stalker's ``screen_<date>.json`` contract (candidates ranked
by technical score) and the OHLCV fixtures behind it - both committed here
under ``fixtures/stockstalker/`` as a snapshot, the same file-contract
pattern the hub uses everywhere else, so this never depends on the two
repos being checked out side by side. Each candidate's daily closes become
monthly returns, regressed against Ken French's factors exactly the way
``factors.regress`` already does for the industry portfolios.

Usage:
    python -m factors.screen_check
    python -m factors.screen_check --model ff3
    python -m factors.screen_check --model ff5mom --diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from factors.capm import run_capm
from factors.diagnostics import run_diagnostics
from factors.kenfrench import load_monthly_factors
from factors.multifactor import FF3_COLUMNS, FF5_MOM_COLUMNS, load_ff5_mom_factors, run_factor_model

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "fixtures" / "stockstalker"
DEFAULT_SCREEN_PATH = FIXTURES_DIR / "screen_2026-09-20.json"
DEFAULT_OHLCV_DIR = FIXTURES_DIR / "ohlcv"

MODEL_FACTOR_COLUMNS = {"capm": ("mkt_rf",), "ff3": FF3_COLUMNS, "ff5mom": FF5_MOM_COLUMNS}

# CAPM's result has no `loadings` dict (just a single beta), so it can't fill
# the contract's `loadings` block -- only the multi-factor models can.
CONTRACT_MODELS = ("ff3", "ff5mom")


def load_screen(path: str | Path = DEFAULT_SCREEN_PATH) -> list[dict]:
    """Read Stock Stalker's screen contract; candidates as ranked there, unchanged."""
    import json

    data = json.loads(Path(path).read_text())
    return data["candidates"]


def ohlcv_filename(ticker: str) -> str:
    """Stock Stalker's fixture naming: ``RELIANCE.NS`` -> ``RELIANCE_NS.csv``."""
    return ticker.replace(".", "_") + ".csv"


def load_monthly_returns(csv_path: str | Path) -> pd.Series:
    """Daily OHLCV close -> monthly returns, indexed by a monthly PeriodIndex.

    Takes each calendar month's last available close and percent-changes
    consecutive months. The final month is dropped whenever the fixture's
    last trading day falls short of that month's real end - otherwise a
    still-in-progress month would price a partial month against a full one
    and call it a return. In this repo's case the Ken French fixture stops
    earlier anyway and the inner join in ``align_excess_returns`` would
    drop it regardless, but the guard keeps this function correct for any
    OHLCV fixture, not just today's.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    monthly_close = df["close"].resample("ME").last()
    if df.index.max() < monthly_close.index[-1]:
        monthly_close = monthly_close.iloc[:-1]
    returns = monthly_close.pct_change().dropna()
    returns.index = pd.PeriodIndex(returns.index, freq="M")
    returns.name = "portfolio"
    return returns


def evaluate_candidate(monthly_returns: pd.Series, model: str):
    """Run the requested model's regression on one candidate's monthly returns."""
    if model == "capm":
        factors_table = load_monthly_factors()
        result = run_capm(monthly_returns, factors_table)
    elif model == "ff3":
        factors_table = load_monthly_factors()
        result = run_factor_model(monthly_returns, factors_table, FF3_COLUMNS)
    elif model == "ff5mom":
        factors_table = load_ff5_mom_factors()
        result = run_factor_model(monthly_returns, factors_table, FF5_MOM_COLUMNS)
    else:
        raise ValueError(f"unknown model '{model}'")
    return result, factors_table


def run_screen_check(
    screen_path: str | Path = DEFAULT_SCREEN_PATH,
    ohlcv_dir: str | Path = DEFAULT_OHLCV_DIR,
    model: str = "capm",
) -> list[dict]:
    """Evaluate every screen candidate; one row per candidate, in screen rank order.

    A candidate whose OHLCV fixture is missing gets an ``error`` key instead
    of a ``result`` - the same "record it, don't hide it" treatment the
    screener itself gives fetch failures, rather than raising and losing
    every other candidate's result.
    """
    rows = []
    for candidate in load_screen(screen_path):
        csv_path = Path(ohlcv_dir) / ohlcv_filename(candidate["ticker"])
        row = dict(candidate)
        if not csv_path.exists():
            row["error"] = f"no OHLCV fixture for {candidate['ticker']} ({csv_path.name})"
            rows.append(row)
            continue
        returns = load_monthly_returns(csv_path)
        try:
            result, factors_table = evaluate_candidate(returns, model)
        except ValueError as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        row["result"] = result
        row["factors_table"] = factors_table
        row["monthly_returns"] = returns
        rows.append(row)
    return rows


def _normalize_ticker(ticker: str) -> str:
    """Stock Stalker's tickers carry a yfinance '.NS' suffix; contract callers may not."""
    return ticker.upper().removesuffix(".NS")


def to_contract(result) -> dict[str, Any]:
    """The ``factor`` block this repo publishes to the spine (v0.2).

    ``result`` must be a ``factors.multifactor.FactorResult`` (FF3 or
    FF5+Mom) -- ``factors.capm.CAPMResult`` has a single beta, not a
    ``loadings`` dict, so CAPM can't fill this contract. The market factor
    is written under the key ``mkt`` (matching NEXT_STEPS.md's committed
    shape), dropping the ``_rf`` suffix this repo's own column carries;
    every other loading keeps its Ken French column name (``smb``, ``hml``,
    ``rmw``, ``cma``, ``mom``) unchanged.
    """
    loadings = {("mkt" if c == "mkt_rf" else c): result.loadings[c] for c in result.factor_columns}
    return {
        "factor": {
            "alpha_annual": result.alpha_annual,
            "alpha_t": result.alpha_t,
            "significant": result.significant,
            "loadings": loadings,
        }
    }


def _print_row(row: dict, model: str, diagnostics: bool) -> None:
    ticker, rank, score = row["ticker"], row["rank"], row["score"]
    if "error" in row:
        print(f"  {rank}. {ticker:<14} technical score {score:.4f}  -- {row['error']}")
        return
    r = row["result"]
    print(
        f"  {rank}. {ticker:<14} technical score {score:.4f}  |  "
        f"alpha(annual) {r.alpha_annual:+.4%}  t={r.alpha_t:+.2f}  "
        f"R2={r.r_squared:.3f}  n={r.n_obs}  significant(5%, OLS)={r.significant}"
    )
    if diagnostics:
        diag = run_diagnostics(row["monthly_returns"], row["factors_table"], MODEL_FACTOR_COLUMNS[model])
        print(
            f"       Newey-West t={diag.alpha_t_nw:+.2f} (lag={diag.nw_lag})  "
            f"significant(NW)={diag.significant_nw}  autocorrelated={diag.autocorrelated}  "
            f"heteroskedastic={diag.heteroskedastic}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screen", default=str(DEFAULT_SCREEN_PATH), help="Stock Stalker screen_<date>.json")
    parser.add_argument("--ohlcv-dir", default=str(DEFAULT_OHLCV_DIR), help="directory of <TICKER>_NS.csv fixtures")
    parser.add_argument("--model", default="capm", choices=["capm", "ff3", "ff5mom"])
    parser.add_argument(
        "--diagnostics", action="store_true", help="add the Newey-West/autocorrelation/heteroskedasticity block"
    )
    parser.add_argument(
        "--contract",
        metavar="PATH",
        default=None,
        help="write the v0.2 factor contract block (see to_contract()) for --contract-ticker "
        "as JSON to this path, for the spine to read as a file -- never as a Python import",
    )
    parser.add_argument(
        "--contract-ticker",
        default=None,
        help="which evaluated candidate's result to write as the contract (requires --contract, "
        "and --model ff3 or ff5mom -- CAPM has no factor loadings)",
    )
    args = parser.parse_args(argv)

    if bool(args.contract) != bool(args.contract_ticker):
        parser.error("--contract and --contract-ticker must be given together")
    if args.contract and args.model not in CONTRACT_MODELS:
        parser.error(f"--contract requires --model to be one of {CONTRACT_MODELS} (CAPM has no factor loadings)")

    rows = run_screen_check(args.screen, args.ohlcv_dir, args.model)

    print(f"Screen check ({args.model.upper()}): does Stock Stalker's technical rank survive factor adjustment?")
    for row in rows:
        _print_row(row, args.model, args.diagnostics)

    evaluated = [r for r in rows if "result" in r]
    significant = [r for r in evaluated if r["result"].significant]
    print()
    if not evaluated:
        print("  no candidate could be evaluated")
    elif significant:
        names = ", ".join(r["ticker"] for r in significant)
        print(f"  {len(significant)}/{len(evaluated)} candidate(s) show significant alpha at 5% (OLS): {names}")
    else:
        print(f"  0/{len(evaluated)} candidates show significant alpha at 5% - the technical edge does not "
              "survive factor adjustment on this universe/window")

    if args.contract:
        target = _normalize_ticker(args.contract_ticker)
        row = next((r for r in evaluated if _normalize_ticker(r["ticker"]) == target), None)
        if row is None:
            parser.error(f"--contract-ticker {args.contract_ticker} not found among this screen's evaluated candidates")
        contract_path = Path(args.contract)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(to_contract(row["result"]), indent=2) + "\n")
        print(f"\nwrote factor contract for {row['ticker']} to {contract_path}")


if __name__ == "__main__":
    main()

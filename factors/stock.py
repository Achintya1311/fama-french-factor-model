"""Day 22 (C1): ticker-agnostic factor regression CLI.

``factors/gulfoillub.py`` (Day 5) hard-wires one ticker: it always reads
``fixtures/stockstalker/ohlcv/GULFOILLUB_NS.csv`` and writes GULFOILLUB's
own contract. C1 needs the same IIMA-factor regression (CAPM / FF3 /
FF3+WML, via the shared ``factors.multifactor.run_factor_model`` path, plus
Newey-West diagnostics) for *any* stock STOCKSTALKER's front door builds a
tear sheet for, not just GULFOILLUB.

This module is that generalisation: ``--ticker`` is a label only (what the
output is printed/written as), ``--prices`` is any OHLCV CSV with
``date``/``close`` columns -- the same file ``stockstalker.data.providers
.local_csv_path`` hands back for a committed fixture or a live yfinance
fetch alike. It shares every underlying function with ``gulfoillub.py``
(``factors.screen_check.load_monthly_returns``, ``factors.iima
.load_iima_factors``, ``factors.multifactor.run_factor_model``) rather than
duplicating them, so the two modules are two thin CLIs over one regression
path and necessarily agree bit-for-bit on the same input (see
``tests/test_stock.py``'s regression test against GULFOILLUB's own
committed fixture).

``factors/gulfoillub.py`` is unchanged and keeps publishing GULFOILLUB's
own pinned contract exactly as before -- this module does not replace it,
it is what STOCKSTALKER's front door now shells out to (via subprocess, as
a file contract -- see ``stockstalker.live_contracts
.generate_factor_contract``) for every other ticker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from factors.diagnostics import run_diagnostics
from factors.iima import load_iima_factors
from factors.multifactor import FactorResult, excess_return_split, print_excess_return_split, run_factor_model
from factors.screen_check import load_monthly_returns

MODEL_FACTOR_COLUMNS: dict[str, tuple[str, ...]] = {
    "capm": ("mf",),
    "ff3": ("mf", "smb", "hml"),
    "ff3wml": ("mf", "smb", "hml", "wml"),
}
MODEL_LABELS = {
    "capm": "CAPM (IIMA MF only)",
    "ff3": "FF3 (IIMA MF/SMB/HML)",
    "ff3wml": "FF3+WML (IIMA MF/SMB/HML/WML)",
}

# Same restriction ``gulfoillub.py`` applies: the contract ships the
# richest spec, so one ticker gets one factor fixture.
CONTRACT_MODEL = "ff3wml"


def load_stock_returns(prices_csv: str | Path) -> pd.Series:
    """A stock's own monthly returns, from any OHLCV CSV with date/close columns.

    Reuses ``factors.screen_check.load_monthly_returns`` unchanged -- it is
    already generic over any such CSV, not specific to the screen's own
    candidates, so no new loader is needed here.
    """
    return load_monthly_returns(prices_csv)


def evaluate(
    model: str,
    prices_csv: str | Path | None = None,
    returns: pd.Series | None = None,
    factors_table: pd.DataFrame | None = None,
) -> tuple[FactorResult, pd.DataFrame]:
    """Run one of ``MODEL_FACTOR_COLUMNS``'s specs against ``prices_csv``.

    ``returns``/``factors_table`` are overridable so tests can hand in a
    synthetic frame without touching disk, the same as ``gulfoillub
    .evaluate``. Exactly one of ``prices_csv`` or ``returns`` must be
    usable: a real run passes ``prices_csv``; a synthetic test passes
    ``returns`` directly and may leave ``prices_csv`` as ``None``.
    """
    if model not in MODEL_FACTOR_COLUMNS:
        raise ValueError(f"unknown model '{model}'; choose one of {sorted(MODEL_FACTOR_COLUMNS)}")
    if returns is None:
        if prices_csv is None:
            raise ValueError("evaluate() needs either prices_csv or returns")
        returns = load_stock_returns(prices_csv)
    factors_table = load_iima_factors() if factors_table is None else factors_table
    result = run_factor_model(returns, factors_table, MODEL_FACTOR_COLUMNS[model])
    return result, factors_table


def to_contract(result: FactorResult) -> dict[str, Any]:
    """The ``factor`` block this module publishes to the spine (v0.2 shape).

    Identical rule to ``gulfoillub.to_contract``: the market factor is
    written under ``mkt`` regardless of IIMA's own column name (``mf``),
    every other loading keeps its own name.
    """
    loadings = {("mkt" if c == "mf" else c): result.loadings[c] for c in result.factor_columns}
    return {
        "factor": {
            "alpha_annual": result.alpha_annual,
            "alpha_t": result.alpha_t,
            "significant": result.significant,
            "loadings": loadings,
            # P98: which factors and which months (additive keys).
            "factor_source": f"IIMA India factors ({'/'.join(c.upper() for c in result.factor_columns)}), monthly",
            "window_start": result.window_start,
            "window_end": result.window_end,
            "n_months": result.n_obs,
        }
    }


def _print_result(ticker: str, model: str, result: FactorResult) -> None:
    label = MODEL_LABELS[model]
    print(f"{label}: {ticker.upper()} vs IIMA factors")
    print(
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    )
    print(f"  months regressed  : {result.n_obs}")
    print(f"  alpha (monthly)   : {result.alpha_monthly:+.4%}")
    print(f"  alpha (annualized): {result.alpha_annual:+.4%}")
    print(f"  alpha t-stat      : {result.alpha_t:+.2f} (OLS, not Newey-West - see --diagnostics)")
    loadings = ", ".join(
        f"{c}={result.loadings[c]:+.3f} (t={result.loadings_t[c]:+.2f})" for c in result.factor_columns
    )
    print(f"  loadings          : {loadings}")
    print(f"  R-squared         : {result.r_squared:.3f}")
    print(f"  significant (5%)  : {result.significant}")


def _print_diagnostics(diag) -> None:
    print("  --- diagnostics ---")
    print(f"  alpha t-stat (NW) : {diag.alpha_t_nw:+.2f} (lag={diag.nw_lag}, vs {diag.alpha_t_ols:+.2f} plain OLS)")
    print(f"  significant (NW)  : {diag.significant_nw}")
    loadings_nw = ", ".join(f"{c}={t:+.2f}" for c, t in diag.loadings_t_nw.items())
    print(f"  loadings t (NW)   : {loadings_nw} (vs plain-OLS t above)")
    print(
        f"  residual autocorr : {diag.autocorrelated} "
        f"(Ljung-Box lag={diag.ljung_box_lag}, stat={diag.ljung_box_stat:.2f}, p={diag.ljung_box_pvalue:.3f})"
    )
    print(
        f"  heteroskedastic   : {diag.heteroskedastic} "
        f"(Breusch-Pagan stat={diag.breusch_pagan_stat:.2f}, p={diag.breusch_pagan_pvalue:.3f})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticker", required=True, help="subject ticker, no .NS suffix (label only)")
    parser.add_argument("--prices", required=True, metavar="PATH", help="OHLCV CSV with date/close columns")
    parser.add_argument(
        "--model", default="ff3wml", choices=sorted(MODEL_FACTOR_COLUMNS), help="which IIMA spec to run"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="add the Newey-West/Ljung-Box/Breusch-Pagan block"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="add the mean monthly excess return split into alpha and factor exposure (P140)",
    )
    parser.add_argument(
        "--contract",
        metavar="PATH",
        default=None,
        help=f"write the v0.2 factor contract (requires --model {CONTRACT_MODEL}) as JSON to this "
        "path, for the spine to read as a file - never as a Python import",
    )
    args = parser.parse_args(argv)

    if args.contract and args.model != CONTRACT_MODEL:
        parser.error(f"--contract requires --model {CONTRACT_MODEL} (the shipped spine spec)")

    try:
        result, factors_table = evaluate(args.model, prices_csv=args.prices)
    except ValueError as exc:
        # D15: an insufficient-data regression (a too-thin inner join, the
        # realistic case for a newly-listed ticker) is a normal, expected
        # outcome -- print the one-line reason and stop, never a full
        # traceback, and never write a fabricated --contract. STOCKSTALKER's
        # generate_factor_contract reads this exact message back from
        # stderr when no contract file lands.
        raise SystemExit(str(exc)) from exc
    _print_result(args.ticker, args.model, result)

    if args.split:
        print_excess_return_split(excess_return_split(load_stock_returns(args.prices), factors_table, result))

    if args.diagnostics:
        diag = run_diagnostics(load_stock_returns(args.prices), factors_table, MODEL_FACTOR_COLUMNS[args.model])
        _print_diagnostics(diag)

    if args.contract:
        contract_path = Path(args.contract)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(to_contract(result), indent=2) + "\n")
        print(f"\nwrote {args.ticker.upper()} factor contract to {contract_path}")


if __name__ == "__main__":
    main()

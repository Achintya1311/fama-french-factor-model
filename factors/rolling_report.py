"""CLI: rolling CAPM beta across all 10 industries (Day 4).

A single full-sample beta (Day 2's CAPM) hides whether a portfolio's market
exposure is stable or drifting - HiTec's beta in 1999 and HiTec's beta in
2009 need not be the same number. This reports a trailing-window beta,
refit every month (``factors.diagnostics.rolling_beta``), for all 10 Ken
French industry portfolios at once and writes the series to
``outputs/rolling_betas.csv``.

Charting this with confidence bands is Day 6's job, not today's - this
script only computes and reports the numbers.

Usage:
    python -m factors.rolling_report
    python -m factors.rolling_report --window 36
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from factors.diagnostics import rolling_beta
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "rolling_betas.csv"


def compute_rolling_betas(window: int) -> pd.DataFrame:
    factors = load_monthly_factors()
    value_weighted = load_monthly_value_weighted()
    series = {
        industry: rolling_beta(value_weighted[industry], factors, window=window)
        for industry in INDUSTRY_COLUMNS
    }
    return pd.DataFrame(series)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="trailing months per beta estimate (default 60, i.e. 5 years)",
    )
    args = parser.parse_args(argv)

    table = compute_rolling_betas(args.window)
    if table.empty:
        raise SystemExit(f"no window of {args.window} months fits inside the available history")

    header = f"{'industry':<8}{'months':>8}{'min':>8}{'current':>9}{'max':>8}"
    print(header)
    print("-" * len(header))
    for industry in INDUSTRY_COLUMNS:
        col = table[industry].dropna()
        print(
            f"{industry:<8}{len(col):>8}{col.min():>8.3f}{col.iloc[-1]:>9.3f}{col.max():>8.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index_label="month")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])

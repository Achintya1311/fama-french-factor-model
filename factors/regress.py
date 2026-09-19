"""CLI: run a factor regression against a Ken French industry portfolio (Day 2).

Only ``--model capm`` is implemented so far. FF3/FF5/momentum (Day 3) will
add more model choices behind the same flag rather than a new command.

Usage:
    python -m factors.regress --portfolio hitec --model capm
"""

from __future__ import annotations

import argparse
import sys

from factors.capm import run_capm
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors


def _resolve_industry(name: str) -> str:
    by_lower = {col.lower(): col for col in INDUSTRY_COLUMNS}
    key = name.strip().lower()
    if key not in by_lower:
        choices = ", ".join(INDUSTRY_COLUMNS)
        raise SystemExit(f"unknown --portfolio '{name}'; choose one of: {choices}")
    return by_lower[key]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--portfolio",
        required=True,
        help=f"one of the 10 Ken French industry portfolios: {', '.join(INDUSTRY_COLUMNS)}",
    )
    parser.add_argument(
        "--model",
        default="capm",
        choices=["capm"],
        help="regression model to run (only 'capm' until Day 3 adds FF3/FF5)",
    )
    args = parser.parse_args(argv)

    industry = _resolve_industry(args.portfolio)
    factors = load_monthly_factors()
    portfolio = load_monthly_value_weighted()[industry]

    result = run_capm(portfolio, factors)

    print(f"CAPM: {industry} value-weighted, vs Mkt-RF")
    print(f"  months regressed  : {result.n_obs} ({portfolio.index.min()} to {portfolio.index.max()})")
    print(f"  alpha (monthly)   : {result.alpha_monthly:+.4%}")
    print(f"  alpha (annualized): {result.alpha_annual:+.4%}")
    print(f"  alpha t-stat      : {result.alpha_t:+.2f} (OLS, not Newey-West - see Day 4)")
    print(f"  beta              : {result.beta:.3f} (t={result.beta_t:+.2f})")
    print(f"  R-squared         : {result.r_squared:.3f}")
    print(f"  significant (5%)  : {result.significant}")


if __name__ == "__main__":
    main(sys.argv[1:])

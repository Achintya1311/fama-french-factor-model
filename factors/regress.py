"""CLI: run a factor regression against a Ken French industry portfolio (Day 2-3).

``--model`` chooses CAPM (Mkt-RF only), FF3 (+ SMB, HML), or FF5+momentum
(+ RMW, CMA, Mom). CAPM keeps its own result type and print format
(``factors.capm``, Day 2, untouched); FF3 and ff5mom share a generic
multi-factor path (``factors.multifactor``, Day 3).

Usage:
    python -m factors.regress --portfolio hitec --model capm
    python -m factors.regress --portfolio hitec --model ff3
    python -m factors.regress --portfolio hitec --model ff5mom
"""

from __future__ import annotations

import argparse
import sys

from factors.capm import run_capm
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import FF3_COLUMNS, FF5_MOM_COLUMNS, load_ff5_mom_factors, run_factor_model

MODEL_LABELS = {"ff3": "FF3", "ff5mom": "FF5+Mom"}


def _resolve_industry(name: str) -> str:
    by_lower = {col.lower(): col for col in INDUSTRY_COLUMNS}
    key = name.strip().lower()
    if key not in by_lower:
        choices = ", ".join(INDUSTRY_COLUMNS)
        raise SystemExit(f"unknown --portfolio '{name}'; choose one of: {choices}")
    return by_lower[key]


def _print_capm(industry: str, portfolio, result) -> None:
    print(f"CAPM: {industry} value-weighted, vs Mkt-RF")
    print(f"  months regressed  : {result.n_obs} ({portfolio.index.min()} to {portfolio.index.max()})")
    print(f"  alpha (monthly)   : {result.alpha_monthly:+.4%}")
    print(f"  alpha (annualized): {result.alpha_annual:+.4%}")
    print(f"  alpha t-stat      : {result.alpha_t:+.2f} (OLS, not Newey-West - see Day 4)")
    print(f"  beta              : {result.beta:.3f} (t={result.beta_t:+.2f})")
    print(f"  R-squared         : {result.r_squared:.3f}")
    print(f"  significant (5%)  : {result.significant}")


def _print_factor_model(model: str, industry: str, portfolio, result) -> None:
    label = MODEL_LABELS[model]
    print(f"{label}: {industry} value-weighted, vs {'/'.join(c.upper() for c in result.factor_columns)}")
    print(f"  months regressed  : {result.n_obs} ({portfolio.index.min()} to {portfolio.index.max()})")
    print(f"  alpha (monthly)   : {result.alpha_monthly:+.4%}")
    print(f"  alpha (annualized): {result.alpha_annual:+.4%}")
    print(f"  alpha t-stat      : {result.alpha_t:+.2f} (OLS, not Newey-West - see Day 4)")
    loadings = ", ".join(
        f"{c}={result.loadings[c]:+.3f} (t={result.loadings_t[c]:+.2f})" for c in result.factor_columns
    )
    print(f"  loadings          : {loadings}")
    print(f"  R-squared         : {result.r_squared:.3f}")
    print(f"  significant (5%)  : {result.significant}")


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
        choices=["capm", "ff3", "ff5mom"],
        help="regression model to run",
    )
    args = parser.parse_args(argv)

    industry = _resolve_industry(args.portfolio)
    portfolio = load_monthly_value_weighted()[industry]

    if args.model == "capm":
        result = run_capm(portfolio, load_monthly_factors())
        _print_capm(industry, portfolio, result)
    elif args.model == "ff3":
        result = run_factor_model(portfolio, load_monthly_factors(), FF3_COLUMNS)
        _print_factor_model(args.model, industry, portfolio, result)
    else:
        result = run_factor_model(portfolio, load_ff5_mom_factors(), FF5_MOM_COLUMNS)
        _print_factor_model(args.model, industry, portfolio, result)


if __name__ == "__main__":
    main(sys.argv[1:])

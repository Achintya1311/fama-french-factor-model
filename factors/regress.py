"""CLI: run a factor regression against a Ken French industry portfolio (Day 2-4).

``--model`` chooses CAPM (Mkt-RF only), FF3 (+ SMB, HML), or FF5+momentum
(+ RMW, CMA, Mom). CAPM keeps its own result type and print format
(``factors.capm``, Day 2, untouched); FF3 and ff5mom share a generic
multi-factor path (``factors.multifactor``, Day 3). ``--diagnostics`` adds a
Day 4 block: Newey-West alpha t-stat alongside the plain-OLS one, plus
Ljung-Box (autocorrelation) and Breusch-Pagan (heteroskedasticity) flags.

Usage:
    python -m factors.regress --portfolio hitec --model capm
    python -m factors.regress --portfolio hitec --model ff3
    python -m factors.regress --portfolio hitec --model ff5mom
    python -m factors.regress --portfolio hitec --model ff5mom --diagnostics
"""

from __future__ import annotations

import argparse
import sys

from factors.capm import run_capm
from factors.diagnostics import run_diagnostics
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import FF3_COLUMNS, FF5_MOM_COLUMNS, load_ff5_mom_factors, run_factor_model

MODEL_LABELS = {"ff3": "FF3", "ff5mom": "FF5+Mom"}
MODEL_FACTOR_COLUMNS = {"capm": ("mkt_rf",), "ff3": FF3_COLUMNS, "ff5mom": FF5_MOM_COLUMNS}


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


def _print_diagnostics(diag) -> None:
    print("  --- diagnostics (Day 4) ---")
    print(f"  alpha t-stat (NW) : {diag.alpha_t_nw:+.2f} (lag={diag.nw_lag}, vs {diag.alpha_t_ols:+.2f} plain OLS)")
    print(f"  significant (NW)  : {diag.significant_nw}")
    print(
        f"  residual autocorr : {diag.autocorrelated} "
        f"(Ljung-Box lag={diag.ljung_box_lag}, stat={diag.ljung_box_stat:.2f}, p={diag.ljung_box_pvalue:.3f}, "
        f"Durbin-Watson={diag.durbin_watson:.2f})"
    )
    print(
        f"  heteroskedastic   : {diag.heteroskedastic} "
        f"(Breusch-Pagan stat={diag.breusch_pagan_stat:.2f}, p={diag.breusch_pagan_pvalue:.3f})"
    )


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
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="add Newey-West t-stats and residual autocorrelation/heteroskedasticity tests",
    )
    args = parser.parse_args(argv)

    industry = _resolve_industry(args.portfolio)
    portfolio = load_monthly_value_weighted()[industry]

    if args.model == "capm":
        factors_table = load_monthly_factors()
        result = run_capm(portfolio, factors_table)
        _print_capm(industry, portfolio, result)
    elif args.model == "ff3":
        factors_table = load_monthly_factors()
        result = run_factor_model(portfolio, factors_table, FF3_COLUMNS)
        _print_factor_model(args.model, industry, portfolio, result)
    else:
        factors_table = load_ff5_mom_factors()
        result = run_factor_model(portfolio, factors_table, FF5_MOM_COLUMNS)
        _print_factor_model(args.model, industry, portfolio, result)

    if args.diagnostics:
        diag = run_diagnostics(portfolio, factors_table, MODEL_FACTOR_COLUMNS[args.model])
        _print_diagnostics(diag)


if __name__ == "__main__":
    main(sys.argv[1:])

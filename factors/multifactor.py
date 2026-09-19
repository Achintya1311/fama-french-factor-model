"""Multi-factor OLS regression: FF3 and FF5+momentum (Day 3).

Generalizes ``factors/capm.py``'s single-factor OLS to an arbitrary set of
factor columns, so FF3 (mkt_rf, smb, hml) and FF5+momentum (mkt_rf, smb,
hml, rmw, cma, mom) share one regression path instead of two near-copies of
each other. ``capm.py`` itself is untouched - it's Day 2 code with its own
tests, and CAPM only ever needs one factor - so ``factors/regress.py`` calls
``run_capm`` for ``--model capm`` and this module for the richer ones.

Standard errors here are still the ordinary OLS ones, not Newey-West - same
caveat as ``capm.py``, same reason (Day 4's job).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

FF3_COLUMNS = ("mkt_rf", "smb", "hml")
FF5_MOM_COLUMNS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")


@dataclass(frozen=True)
class FactorResult:
    n_obs: int
    factor_columns: tuple[str, ...]
    alpha_monthly: float
    alpha_annual: float
    alpha_t: float
    loadings: dict[str, float]
    loadings_t: dict[str, float]
    r_squared: float
    significant: bool  # |alpha_t| > 1.96, i.e. non-robust 5% two-sided test


def align_excess_returns(
    portfolio_monthly: pd.Series,
    factors_monthly: pd.DataFrame,
    factor_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Inner-join a portfolio's monthly returns with the factor table.

    Returns a frame indexed by the overlapping months with an ``excess``
    column (portfolio return minus RF) plus one column per entry in
    ``factor_columns``. Raises ``ValueError`` if the two series share no
    months, or if ``factors_monthly`` is missing ``rf`` or any requested
    factor column.
    """
    missing = [c for c in (*factor_columns, "rf") if c not in factors_monthly.columns]
    if missing:
        raise ValueError(f"factors_monthly is missing columns: {missing}")
    joined = pd.concat(
        [portfolio_monthly.rename("portfolio"), factors_monthly[[*factor_columns, "rf"]]],
        axis=1,
        join="inner",
    )
    if joined.empty:
        raise ValueError("portfolio and factor series share no overlapping months")
    out = pd.DataFrame({"excess": joined["portfolio"] - joined["rf"]})
    for col in factor_columns:
        out[col] = joined[col]
    return out


def run_factor_model(
    portfolio_monthly: pd.Series,
    factors_monthly: pd.DataFrame,
    factor_columns: tuple[str, ...],
) -> FactorResult:
    """Fit the multi-factor regression and return alpha/loadings/t-stats/R-squared.

    ``alpha_annual`` compounds the monthly alpha, matching ``run_capm``.
    """
    aligned = align_excess_returns(portfolio_monthly, factors_monthly, factor_columns)
    X = sm.add_constant(aligned[list(factor_columns)])
    model = sm.OLS(aligned["excess"], X).fit()

    alpha_monthly = float(model.params["const"])
    alpha_t = float(model.tvalues["const"])
    return FactorResult(
        n_obs=int(model.nobs),
        factor_columns=tuple(factor_columns),
        alpha_monthly=alpha_monthly,
        alpha_annual=(1.0 + alpha_monthly) ** 12 - 1.0,
        alpha_t=alpha_t,
        loadings={c: float(model.params[c]) for c in factor_columns},
        loadings_t={c: float(model.tvalues[c]) for c in factor_columns},
        r_squared=float(model.rsquared),
        significant=abs(alpha_t) > 1.96,
    )


def load_ff5_mom_factors() -> pd.DataFrame:
    """Return the merged FF5 + momentum monthly factor table.

    Inner-joins ``factors.five_factor.load_monthly_five_factors()`` (mkt_rf,
    smb, hml, rmw, cma, rf) with ``factors.momentum.load_monthly_momentum()``
    (mom) on month, since French ships momentum as a separate download from
    FF5 rather than a seventh column in the same file.
    """
    from factors.five_factor import load_monthly_five_factors
    from factors.momentum import load_monthly_momentum

    five = load_monthly_five_factors()
    mom = load_monthly_momentum()
    merged = five.join(mom, how="inner")
    if merged.empty:
        raise ValueError("FF5 and momentum series share no overlapping months")
    return merged

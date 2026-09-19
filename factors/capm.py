"""CAPM baseline regression (Day 2).

Regresses a portfolio's excess return on the market factor (Mkt-RF) by
plain OLS: ``r_p - r_f = alpha + beta * (r_m - r_f) + e``.

Standard errors here are the ordinary OLS ones, not Newey-West - the repo's
own traps note that overlapping/serially-correlated monthly returns inflate
t-stats, and that correction is Day 4's job (``Diagnostics``). Until then,
``alpha_t`` should be read as optimistic, not final.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class CAPMResult:
    n_obs: int
    alpha_monthly: float
    alpha_annual: float
    alpha_t: float
    beta: float
    beta_t: float
    r_squared: float
    significant: bool  # |alpha_t| > 1.96, i.e. non-robust 5% two-sided test


def align_excess_returns(
    portfolio_monthly: pd.Series, factors_monthly: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join a portfolio's monthly returns with the factor table.

    Returns a frame indexed by the overlapping months with columns
    ``excess`` (portfolio return minus RF) and ``mkt_rf``. Raises
    ``ValueError`` if the two series share no months.
    """
    joined = pd.concat(
        [portfolio_monthly.rename("portfolio"), factors_monthly[["mkt_rf", "rf"]]],
        axis=1,
        join="inner",
    )
    if joined.empty:
        raise ValueError("portfolio and factor series share no overlapping months")
    return pd.DataFrame(
        {
            "excess": joined["portfolio"] - joined["rf"],
            "mkt_rf": joined["mkt_rf"],
        }
    )


def run_capm(portfolio_monthly: pd.Series, factors_monthly: pd.DataFrame) -> CAPMResult:
    """Fit the CAPM regression and return alpha/beta/t-stats/R-squared.

    ``alpha_annual`` compounds the monthly alpha, ``(1 + alpha_monthly)**12
    - 1``, matching how the factor returns themselves are decimal monthly
    figures.
    """
    aligned = align_excess_returns(portfolio_monthly, factors_monthly)
    X = sm.add_constant(aligned["mkt_rf"])
    model = sm.OLS(aligned["excess"], X).fit()

    alpha_monthly = float(model.params["const"])
    alpha_t = float(model.tvalues["const"])
    return CAPMResult(
        n_obs=int(model.nobs),
        alpha_monthly=alpha_monthly,
        alpha_annual=(1.0 + alpha_monthly) ** 12 - 1.0,
        alpha_t=alpha_t,
        beta=float(model.params["mkt_rf"]),
        beta_t=float(model.tvalues["mkt_rf"]),
        r_squared=float(model.rsquared),
        significant=abs(alpha_t) > 1.96,
    )

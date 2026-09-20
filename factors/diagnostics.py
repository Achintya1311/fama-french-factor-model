"""Regression diagnostics for Day 2-3's factor models (Day 4).

Newey-West (HAC) standard errors, a Ljung-Box test for residual
autocorrelation, and a Breusch-Pagan test for heteroskedasticity, all built
on top of ``factors.multifactor.align_excess_returns`` - the same alignment
CAPM, FF3, and FF5+Mom already share. Passing ``factor_columns=("mkt_rf",)``
runs these diagnostics on the CAPM specification without touching
``capm.py``'s own dataclass; ``FF3_COLUMNS``/``FF5_MOM_COLUMNS`` cover the
other two.

Why this module exists: ``capm.py`` and ``multifactor.py`` both report
plain-OLS t-stats and say so in their own docstrings - overlapping monthly
returns are serially correlated, which inflates plain-OLS t-stats and can
turn a noisy alpha into a false "significant". Newey-West doesn't fix the
correlation, it corrects the standard error for it, so the same alpha now
gets an honest t-stat instead of an optimistic one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from factors.multifactor import align_excess_returns


def newey_west_lag(n_obs: int) -> int:
    """Newey & West's (1994) automatic bandwidth rule: floor(4*(n/100)**(2/9)).

    Grows slowly with sample size (12 lags at n=100, 21 at n=1200) rather
    than using a single fixed lag for every portfolio regressed here, some
    of which run on 240 months (Day 3's synthetic tests) and some on 1,201
    (the full Ken French history). Floored at 1 so a tiny sample never asks
    for zero lags.
    """
    return max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))


@dataclass(frozen=True)
class DiagnosticsResult:
    n_obs: int
    factor_columns: tuple[str, ...]
    nw_lag: int
    alpha_t_ols: float
    alpha_t_nw: float
    significant_ols: bool  # |alpha_t_ols| > 1.96, the plain-OLS reading
    significant_nw: bool  # |alpha_t_nw| > 1.96, the Newey-West-corrected reading
    loadings_t_nw: dict[str, float]
    ljung_box_lag: int
    ljung_box_stat: float
    ljung_box_pvalue: float
    autocorrelated: bool  # ljung_box_pvalue < 0.05: residuals aren't white noise
    breusch_pagan_stat: float
    breusch_pagan_pvalue: float
    heteroskedastic: bool  # breusch_pagan_pvalue < 0.05: residual variance isn't constant
    durbin_watson: float  # ~2.0 = no first-order autocorrelation, <2 = positive


def run_diagnostics(
    portfolio_monthly: pd.Series,
    factors_monthly: pd.DataFrame,
    factor_columns: tuple[str, ...],
    ljung_box_lag: int = 12,
) -> DiagnosticsResult:
    """Fit the same OLS spec as ``capm.run_capm``/``multifactor.run_factor_model``
    and report what plain OLS alone doesn't: a HAC-robust alpha t-stat plus
    tests for the two conditions (serial correlation, heteroskedasticity)
    that make the plain one unreliable in the first place.

    ``ljung_box_lag=12`` tests residual autocorrelation up to a year back -
    monthly data, so a year is the natural horizon for "does last month's
    miss predict this month's".
    """
    aligned = align_excess_returns(portfolio_monthly, factors_monthly, factor_columns)
    X = sm.add_constant(aligned[list(factor_columns)])
    y = aligned["excess"]

    ols = sm.OLS(y, X).fit()
    lag = newey_west_lag(int(ols.nobs))
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True})

    lb = acorr_ljungbox(ols.resid, lags=[ljung_box_lag], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    bp_stat, bp_pvalue, _, _ = het_breuschpagan(ols.resid, ols.model.exog)

    alpha_t_ols = float(ols.tvalues["const"])
    alpha_t_nw = float(hac.tvalues["const"])
    return DiagnosticsResult(
        n_obs=int(ols.nobs),
        factor_columns=tuple(factor_columns),
        nw_lag=lag,
        alpha_t_ols=alpha_t_ols,
        alpha_t_nw=alpha_t_nw,
        significant_ols=abs(alpha_t_ols) > 1.96,
        significant_nw=abs(alpha_t_nw) > 1.96,
        loadings_t_nw={c: float(hac.tvalues[c]) for c in factor_columns},
        ljung_box_lag=ljung_box_lag,
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_pvalue,
        autocorrelated=lb_pvalue < 0.05,
        breusch_pagan_stat=float(bp_stat),
        breusch_pagan_pvalue=float(bp_pvalue),
        heteroskedastic=float(bp_pvalue) < 0.05,
        durbin_watson=float(durbin_watson(ols.resid)),
    )


def rolling_beta(
    portfolio_monthly: pd.Series,
    factors_monthly: pd.DataFrame,
    factor_column: str = "mkt_rf",
    window: int = 60,
) -> pd.Series:
    """Trailing ``window``-month single-factor beta, refit at every month.

    Windows shorter than ``window`` are dropped rather than partially fit -
    a beta from 12 months of data on a "60-month rolling beta" chart would
    be a different, noisier statistic wearing the same label. Returned
    series is indexed by each window's last month.
    """
    aligned = align_excess_returns(portfolio_monthly, factors_monthly, (factor_column,))
    if len(aligned) < window:
        return pd.Series([], dtype=float, name=f"beta_{factor_column}").rename_axis("month")
    X = sm.add_constant(aligned[[factor_column]])
    fitted = RollingOLS(aligned["excess"], X, window=window, min_nobs=window).fit()
    betas = fitted.params[factor_column].dropna()
    betas.name = f"beta_{factor_column}"
    betas.index.name = "month"
    return betas

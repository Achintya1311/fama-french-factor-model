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

# D15: a regression with too few months relative to its own parameter count
# (intercept + each factor column) is rank-deficient or exactly/near-exactly
# determined - a fabricated "perfect fit" (R-squared=1.0, t-stats collapsing
# to 0.0/0.0), not a real finding. Requiring at least 3x the parameter count
# keeps every case this repo already treats as merely "underpowered" (Day
# 5-7's 22-month Ken French screen check passes with a 1-month margin for
# FF5+Mom's 7 parameters) while rejecting the few-month cases a newly-listed
# ticker's "any stock" tear sheet (C1) can otherwise hit.
MIN_OBS_PARAM_MULTIPLE = 3


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
    window_start: str | None = None  # P98: first regressed month, e.g. "2014-08"
    window_end: str | None = None  # P98: last regressed month


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
    n_params = len(factor_columns) + 1
    min_obs = MIN_OBS_PARAM_MULTIPLE * n_params
    if len(aligned) < min_obs:
        raise ValueError(
            f"too few observations for a factor regression: n_obs={len(aligned)} against "
            f"{len(factor_columns)} factor columns ({n_params} parameters incl. intercept) "
            f"needs at least {min_obs} months ({MIN_OBS_PARAM_MULTIPLE}x the parameter count)"
        )
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
        window_start=str(aligned.index[0]),
        window_end=str(aligned.index[-1]),
    )


def excess_return_split(
    portfolio_monthly: pd.Series,
    factors_monthly: pd.DataFrame,
    result: FactorResult,
) -> dict:
    """Split the mean monthly excess return into alpha and factor exposure (P140).

    With an intercept, OLS residuals average zero, so on the regressed months
    mean(excess) = alpha + sum(loading x mean factor) exactly. Returns
    ``mean_excess``, ``alpha``, ``exposure`` and ``by_factor`` (one term per
    factor), all monthly, arithmetic.
    """
    aligned = align_excess_returns(portfolio_monthly, factors_monthly, result.factor_columns)
    by_factor = {c: result.loadings[c] * float(aligned[c].mean()) for c in result.factor_columns}
    return {
        "mean_excess": float(aligned["excess"].mean()),
        "alpha": result.alpha_monthly,
        "exposure": float(sum(by_factor.values())),
        "by_factor": by_factor,
    }


def print_excess_return_split(split: dict) -> None:
    print("  --- mean excess return split (monthly, arithmetic; parts add up exactly under OLS) ---")
    print(f"  mean excess return (monthly): {split['mean_excess']:+.2%}")
    print(f"  alpha                       : {split['alpha']:+.2%}")
    print(f"  factor exposure             : {split['exposure']:+.2%}")
    terms = ", ".join(f"{c}={v:+.2%}" for c, v in split["by_factor"].items())
    print(f"    by factor (loading x mean): {terms}")


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


def alpha_change_attribution(
    portfolio_monthly: pd.Series,
    base_factors: pd.DataFrame,
    base_columns: tuple[str, ...],
    rich_factors: pd.DataFrame,
    rich_columns: tuple[str, ...],
) -> dict:
    """P128 (CRITIQUES.md): split the monthly alpha change between two models by factor.

    OLS with an intercept gives ``alpha = mean(excess) - sum(beta_k * mean(f_k))``
    on each model's own sample, so ``alpha_rich - alpha_base`` is exactly
    ``sum_k (beta_base_k * mean_base(f_k) - beta_rich_k * mean_rich(f_k))``
    (a factor absent from a model has beta 0 there) plus ``residual``, the
    difference in mean excess return between the two aligned samples (zero
    when both share months and RF). Each model's own factor series is used,
    so a factor whose definition differs between files (FF5's SMB vs FF3's)
    shows up inside its contribution rather than as a residual.
    """
    base = run_factor_model(portfolio_monthly, base_factors, base_columns)
    rich = run_factor_model(portfolio_monthly, rich_factors, rich_columns)
    base_aligned = align_excess_returns(portfolio_monthly, base_factors, base_columns)
    rich_aligned = align_excess_returns(portfolio_monthly, rich_factors, rich_columns)
    contributions = {}
    for column in dict.fromkeys((*base_columns, *rich_columns)):
        base_term = base.loadings.get(column, 0.0) * (base_aligned[column].mean() if column in base_columns else 0.0)
        rich_term = rich.loadings.get(column, 0.0) * (rich_aligned[column].mean() if column in rich_columns else 0.0)
        contributions[column] = float(base_term - rich_term)
    return {
        "alpha_change": rich.alpha_monthly - base.alpha_monthly,
        "contributions": contributions,
        "residual": float(rich_aligned["excess"].mean() - base_aligned["excess"].mean()),
        "base_loadings": base.loadings,
        "rich_loadings": rich.loadings,
        "n_obs": (base.n_obs, rich.n_obs),
    }

"""Panel-alpha cross-section test (R17): is there alpha across the NSE F&O
universe, not just in GULFOILLUB's own single-stock series?

Pre-registered in ``docs/PREREG_panel_alpha.md`` before this module was
written or any regression run against the real panel. Reuses this repo's
existing building blocks rather than re-implementing them:

- ``factors.multifactor.run_factor_model`` -- the same FF3+WML OLS spec
  ``factors.gulfoillub``/``factors.stock`` already run per stock.
- ``factors.diagnostics.run_diagnostics`` -- Newey-West alpha t-stat,
  reported alongside OLS for every stock, the same convention as Day 5.
- ``factors.multiple_testing.two_sided_normal_pvalue``/``benjamini_hochberg``
  -- the existing FDR-control implementation, called here at the 10% level
  R17 specifies rather than this repo's other 5%-default usages.

Two things are new here, because nothing in this repo needed them before
a *panel* (many assets) existed: the GRS joint test of zero alpha
(Gibbons, Ross & Shanken, 1989) and a minimum-detectable-alpha power
statement.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from factors.diagnostics import run_diagnostics
from factors.iima import load_iima_factors
from factors.multifactor import FactorResult, run_factor_model
from factors.multiple_testing import benjamini_hochberg, two_sided_normal_pvalue
from factors.screen_check import load_monthly_returns
from factors.universe import load_fo_stock_tickers

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL_OHLCV_DIR = ROOT / "fixtures" / "panel_ohlcv"

FF3WML_COLUMNS: tuple[str, ...] = ("mf", "smb", "hml", "wml")

# docs/PREREG_panel_alpha.md: a stock below this many overlapping months
# with IIMA's factor history is dropped from the per-stock panel entirely
# (not just a looser anti-degenerate-fit floor -- factors/multifactor.py's
# own D15 floor for FF3+WML is 15 months; this is a *power* requirement).
MIN_PANEL_MONTHS = 60

# docs/PREREG_panel_alpha.md: the per-stock panel floor (R17's own
# instruction). The GRS common-window subset is reported separately and
# may be smaller -- see module docstring and run_panel's own docstring.
MIN_PANEL_STOCKS = 150


def panel_ohlcv_path(ticker: str, fixtures_dir: Path = DEFAULT_PANEL_OHLCV_DIR) -> Path:
    return fixtures_dir / f"{ticker.replace('.', '_')}.csv"


@dataclass(frozen=True)
class PanelStockResult:
    ticker: str
    n_obs: int
    alpha_monthly: float
    alpha_annual: float
    alpha_t_ols: float
    alpha_t_nw: float
    r_squared: float
    se_alpha_monthly: float


@dataclass(frozen=True)
class PanelBuildResult:
    results: list[PanelStockResult]
    skipped: dict[str, str]  # ticker -> reason


def mle_covariance(residuals: np.ndarray) -> np.ndarray:
    """The maximum-likelihood (divide-by-T) covariance matrix of a T x N
    residual matrix, columns demeaned first.

    GRS (1989) is defined using the MLE covariance, not the usual
    divide-by-(T-1) sample covariance -- the two don't cancel in the GRS
    ratio (Sigma, an N x N matrix, and Omega, a K x K matrix, are inverted
    separately, with different T-1 vs T biases if the convention were
    mixed up), so this is kept as its own explicit, tested function rather
    than a bare ``np.cov`` call at each use site.
    """
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    t = residuals.shape[0]
    return (centered.T @ centered) / t


@dataclass(frozen=True)
class GRSResult:
    stat: float
    p_value: float
    df1: int
    df2: int
    n_assets: int
    n_periods: int


def grs_test(
    alphas: np.ndarray,
    resid_cov: np.ndarray,
    factor_means: np.ndarray,
    factor_cov: np.ndarray,
    T: int,
) -> GRSResult:
    """Gibbons, Ross & Shanken's (1989) joint test that every asset's alpha
    is zero.

    ``GRS = ((T-N-K)/N) * (alpha' @ Sigma^-1 @ alpha) / (1 + mu_f' @ Omega^-1 @ mu_f)``,
    distributed ``F(N, T-N-K)`` under the null. ``resid_cov``/``factor_cov``
    must be the MLE (divide-by-T) covariance matrices (``mle_covariance``),
    not numpy's divide-by-(T-1) default.
    """
    alphas = np.asarray(alphas, dtype=float)
    n = alphas.shape[0]
    k = np.asarray(factor_means, dtype=float).shape[0]
    df2 = T - n - k
    if df2 <= 0:
        raise ValueError(
            f"non-positive degrees of freedom: T={T}, N={n}, K={k} gives T-N-K={df2}; "
            "need more time periods, fewer assets, or fewer factors"
        )

    resid_cov = np.asarray(resid_cov, dtype=float)
    factor_cov = np.asarray(factor_cov, dtype=float)
    factor_means = np.asarray(factor_means, dtype=float)

    quad_alpha = float(alphas @ np.linalg.solve(resid_cov, alphas))
    quad_factor = float(factor_means @ np.linalg.solve(factor_cov, factor_means))

    stat = ((T - n - k) / n) * (quad_alpha / (1.0 + quad_factor))
    p_value = float(stats.f.sf(stat, n, df2))
    return GRSResult(stat=stat, p_value=p_value, df1=n, df2=df2, n_assets=n, n_periods=T)


def minimum_detectable_alpha_monthly(se_alpha_monthly: float, power: float = 0.80, alpha_level: float = 0.05) -> float:
    """Smallest monthly alpha detectable at ``power`` with a two-sided test
    at ``alpha_level``, given this stock's own fitted ``se(alpha_hat)``.

    Standard minimum-detectable-effect formula:
    ``MDE = (z_(1-alpha/2) + z_power) * se``.
    """
    z_alpha = float(stats.norm.ppf(1.0 - alpha_level / 2.0))
    z_power = float(stats.norm.ppf(power))
    return (z_alpha + z_power) * se_alpha_monthly


def minimum_detectable_alpha_annual(se_alpha_monthly: float, power: float = 0.80, alpha_level: float = 0.05) -> float:
    """``minimum_detectable_alpha_monthly`` compounded to an annual figure,
    the same transform ``FactorResult.alpha_annual`` already uses for a
    point estimate."""
    mde_monthly = minimum_detectable_alpha_monthly(se_alpha_monthly, power, alpha_level)
    return (1.0 + mde_monthly) ** 12 - 1.0


@dataclass(frozen=True)
class BHCorrectionResult:
    fdr: float
    p_values: dict[str, float]
    rejected: dict[str, bool]
    n_rejected: int
    n_raw_significant: int  # |alpha_t_ols| > 1.96, no correction


def run_bh_correction(results: list[PanelStockResult], fdr: float = 0.10) -> BHCorrectionResult:
    """Benjamini-Hochberg FDR control over the panel's per-stock alpha
    p-values, reusing ``factors.multiple_testing.benjamini_hochberg``
    exactly -- not a reimplementation -- at the 10%-FDR level R17
    specifies (this repo's other call sites use 5%)."""
    tickers = [r.ticker for r in results]
    p_values = [two_sided_normal_pvalue(r.alpha_t_ols) for r in results]
    rejected_flags = benjamini_hochberg(p_values, alpha=fdr)
    n_raw_significant = sum(1 for r in results if abs(r.alpha_t_ols) > 1.96)
    return BHCorrectionResult(
        fdr=fdr,
        p_values=dict(zip(tickers, p_values)),
        rejected=dict(zip(tickers, rejected_flags)),
        n_rejected=sum(rejected_flags),
        n_raw_significant=n_raw_significant,
    )


def _load_panel_returns(
    tickers: list[str], fixtures_dir: Path = DEFAULT_PANEL_OHLCV_DIR
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load each ticker's own monthly-return series; a ticker with no
    committed fixture (or an unreadable one) is skipped and logged, not
    allowed to abort the whole panel build."""
    returns: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
    for ticker in tickers:
        path = panel_ohlcv_path(ticker, fixtures_dir)
        if not path.exists():
            skipped[ticker] = f"no committed panel fixture at {path}"
            continue
        try:
            returns[ticker] = load_monthly_returns(path)
        except Exception as exc:  # a malformed fixture must not abort the panel
            skipped[ticker] = f"could not load {path}: {exc}"
    return returns, skipped


def run_panel(
    tickers: list[str] | None = None,
    fixtures_dir: Path = DEFAULT_PANEL_OHLCV_DIR,
    factors_table: pd.DataFrame | None = None,
    min_months: int = MIN_PANEL_MONTHS,
) -> PanelBuildResult:
    """Run the FF3+WML regression for every panel stock meeting the
    ``min_months``-overlap floor. Does not attempt the GRS common window --
    see ``grs_common_window_panel`` -- since BH/R-squared/the power
    statement have no common-window requirement and dropping stocks to one
    would understate the panel this test is actually reporting on.
    """
    tickers = load_fo_stock_tickers() if tickers is None else tickers
    factors_table = load_iima_factors() if factors_table is None else factors_table

    raw_returns, skipped = _load_panel_returns(tickers, fixtures_dir)

    results: list[PanelStockResult] = []
    for ticker, monthly_returns in raw_returns.items():
        try:
            fit = run_factor_model(monthly_returns, factors_table, FF3WML_COLUMNS)
        except ValueError as exc:
            skipped[ticker] = str(exc)
            continue
        if fit.n_obs < min_months:
            skipped[ticker] = f"only {fit.n_obs} overlapping months, need >= {min_months}"
            continue
        diag = run_diagnostics(monthly_returns, factors_table, FF3WML_COLUMNS)
        se_alpha_monthly = fit.alpha_monthly / fit.alpha_t if fit.alpha_t != 0 else float("nan")
        results.append(
            PanelStockResult(
                ticker=ticker,
                n_obs=fit.n_obs,
                alpha_monthly=fit.alpha_monthly,
                alpha_annual=fit.alpha_annual,
                alpha_t_ols=fit.alpha_t,
                alpha_t_nw=diag.alpha_t_nw,
                r_squared=fit.r_squared,
                se_alpha_monthly=se_alpha_monthly,
            )
        )
    return PanelBuildResult(results=results, skipped=skipped)


@dataclass(frozen=True)
class GRSPanelResult:
    grs: GRSResult | None
    tickers: list[str]
    window_start: str
    window_end: str
    infeasible_reason: str | None = None


def grs_common_window_panel(
    tickers: list[str],
    fixtures_dir: Path = DEFAULT_PANEL_OHLCV_DIR,
    factors_table: pd.DataFrame | None = None,
) -> GRSPanelResult | None:
    """Restrict to the subset of panel stocks with full price coverage over
    one shared monthly window (common start = the latest of every included
    stock's own first month; common end = IIMA's own last month), refit
    FF3+WML over exactly that window for each, and run the GRS joint test
    on the resulting alpha vector and residual covariance.

    Returns ``None`` if fewer than ``K+2`` assets survive the common-window
    filter (the bare minimum for ``Sigma`` to be invertible) -- per
    docs/PREREG_panel_alpha.md, that shortfall is itself a finding, not
    patched by loosening the window after the fact.

    ``tickers`` must already be the per-stock panel (``run_panel``'s own
    60-month-floor survivors), not the raw universe -- a newly-listed
    stock with only a handful of months would otherwise push the common
    start date to just a few months ago and collapse T for every stock,
    which is a caller bug, not the common-window test this function is
    pre-registered to run.
    """
    factors_table = load_iima_factors() if factors_table is None else factors_table
    raw_returns, _ = _load_panel_returns(tickers, fixtures_dir)
    if not raw_returns:
        return None

    factor_end = factors_table.index.max()
    # Every stock's own earliest month with a committed price bar; the
    # common window can start no earlier than the latest of these.
    starts = [series.index.min() for series in raw_returns.values()]
    window_start = max(starts)
    window_end = factor_end

    aligned_factors = factors_table.loc[window_start:window_end]
    if aligned_factors.empty:
        return None

    included: list[str] = []
    alpha_vec: list[float] = []
    resid_matrix_cols: list[np.ndarray] = []
    for ticker, series in raw_returns.items():
        window_returns = series.reindex(aligned_factors.index)
        if window_returns.isna().any():
            continue  # this stock has a gap inside the common window - excluded, not filled
        joined = pd.concat(
            [window_returns.rename("portfolio"), aligned_factors[[*FF3WML_COLUMNS, "rf"]]],
            axis=1,
            join="inner",
        )
        if len(joined) != len(aligned_factors):
            continue
        excess = joined["portfolio"] - joined["rf"]
        import statsmodels.api as sm

        X = sm.add_constant(joined[list(FF3WML_COLUMNS)])
        fit = sm.OLS(excess, X).fit()
        included.append(ticker)
        alpha_vec.append(float(fit.params["const"]))
        resid_matrix_cols.append(fit.resid.to_numpy())

    k = len(FF3WML_COLUMNS)
    if len(included) < k + 2:
        return None

    T = len(aligned_factors)
    n = len(included)
    if T - n - k <= 0:
        # The common window is bounded by the latest-starting panel stock's
        # own first month, and a cross-section this wide (N) can easily
        # exceed that window's own length (T) once every included stock
        # must share it -- Sigma (N x N) estimated from T < N observations
        # is then rank-deficient and GRS cannot be computed as specified.
        # Per docs/PREREG_panel_alpha.md this is reported honestly, not
        # patched by dropping stocks to force T > N after seeing the result.
        return GRSPanelResult(
            grs=None,
            tickers=included,
            window_start=str(window_start),
            window_end=str(window_end),
            infeasible_reason=(
                f"T={T} common-window months is not enough for N={n} assets and K={k} "
                f"factors (need T > N+K = {n + k}); Sigma would be rank-deficient"
            ),
        )

    resid_matrix = np.column_stack(resid_matrix_cols)  # T x N
    resid_cov = mle_covariance(resid_matrix)
    factor_values = aligned_factors[list(FF3WML_COLUMNS)].to_numpy()
    factor_cov = mle_covariance(factor_values)
    factor_means = factor_values.mean(axis=0)

    grs = grs_test(np.array(alpha_vec), resid_cov, factor_means, factor_cov, T)
    return GRSPanelResult(
        grs=grs,
        tickers=included,
        window_start=str(window_start),
        window_end=str(window_end),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default=str(DEFAULT_PANEL_OHLCV_DIR))
    parser.add_argument("--min-months", type=int, default=MIN_PANEL_MONTHS)
    parser.add_argument("--fdr", type=float, default=0.10)
    args = parser.parse_args(argv)

    fixtures_dir = Path(args.fixtures_dir)
    tickers = load_fo_stock_tickers()

    build = run_panel(tickers, fixtures_dir=fixtures_dir, min_months=args.min_months)
    n = len(build.results)
    print(f"panel: {n}/{len(tickers)} stocks meet the {args.min_months}-month floor "
          f"({len(build.skipped)} skipped)")
    if n == 0:
        print("no panel data available - nothing to test", file=sys.stderr)
        return 1

    bh = run_bh_correction(build.results, fdr=args.fdr)
    print(f"Benjamini-Hochberg @ {args.fdr:.0%} FDR: {bh.n_rejected}/{n} alphas survive "
          f"(vs {bh.n_raw_significant}/{n} raw-significant at 5%, uncorrected)")

    r_squareds = [r.r_squared for r in build.results]
    print(f"R-squared across panel: mean={np.mean(r_squareds):.3f} median={np.median(r_squareds):.3f}")

    mdes = [minimum_detectable_alpha_annual(r.se_alpha_monthly) for r in build.results if r.se_alpha_monthly == r.se_alpha_monthly]
    if mdes:
        print(f"minimum detectable alpha (80% power, annualized): "
              f"median={np.median(mdes):.4f} mean={np.mean(mdes):.4f}")

    panel_tickers = [r.ticker for r in build.results]
    grs_panel = grs_common_window_panel(panel_tickers, fixtures_dir=fixtures_dir)
    if grs_panel is None:
        print("GRS common-window panel too small to compute (need >= K+2 assets "
              "with full coverage over the shared window)", file=sys.stderr)
    elif grs_panel.grs is None:
        print(f"GRS not computable: N={len(grs_panel.tickers)} assets over window "
              f"{grs_panel.window_start} to {grs_panel.window_end} -- {grs_panel.infeasible_reason}",
              file=sys.stderr)
    else:
        g = grs_panel.grs
        print(f"GRS joint test (N={g.n_assets}, T={g.n_periods}, window "
              f"{grs_panel.window_start} to {grs_panel.window_end}): "
              f"stat={g.stat:.4f}, p={g.p_value:.4f}, "
              f"{'REJECTS' if g.p_value < 0.05 else 'does not reject'} H0: all alphas = 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

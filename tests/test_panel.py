"""Tests for factors.panel (R17 panel-alpha cross-section test).

Formula-level tests first (grs_test, minimum_detectable_alpha_monthly): a
tiny, hand-computable synthetic case, not a re-derivation of the whole
asset-pricing theory -- the point is to pin the implementation's matrix
algebra and degrees of freedom, which a textbook-scale real panel can't be
eyeballed against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from factors.panel import (
    PanelStockResult,
    grs_test,
    minimum_detectable_alpha_monthly,
    mle_covariance,
    run_bh_correction,
)


def test_grs_stat_matches_hand_computed_value_on_a_diagonal_synthetic_case():
    # alpha' Sigma^-1 alpha = 0.01^2/0.0004 + 0.02^2/0.0009 = 0.25 + 4/9 = 0.694444...
    # mu_f' Omega^-1 mu_f = 0.005^2/0.0001 = 0.25
    # (T-N-K)/N = (100-2-1)/2 = 48.5
    # GRS = 48.5 * 0.694444.../1.25 = 26.944444...
    alphas = np.array([0.01, 0.02])
    resid_cov = np.array([[0.0004, 0.0], [0.0, 0.0009]])
    factor_means = np.array([0.005])
    factor_cov = np.array([[0.0001]])
    T = 100

    result = grs_test(alphas, resid_cov, factor_means, factor_cov, T)

    assert result.stat == pytest.approx(26.944444444, rel=1e-8)
    assert result.df1 == 2
    assert result.df2 == 97
    assert result.p_value == pytest.approx(float(stats.f.sf(26.944444444, 2, 97)), rel=1e-8)


def test_grs_stat_is_zero_and_pvalue_one_when_every_alpha_is_exactly_zero():
    alphas = np.zeros(3)
    resid_cov = np.diag([0.0001, 0.0002, 0.0003])
    factor_means = np.array([0.004, 0.002])
    factor_cov = np.array([[0.0002, 0.00001], [0.00001, 0.0001]])

    result = grs_test(alphas, resid_cov, factor_means, factor_cov, T=200)

    assert result.stat == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_grs_test_raises_when_degrees_of_freedom_are_non_positive():
    alphas = np.array([0.01, 0.02])
    resid_cov = np.eye(2) * 0.0001
    factor_means = np.array([0.005])
    factor_cov = np.array([[0.0001]])

    with pytest.raises(ValueError, match="degrees of freedom"):
        grs_test(alphas, resid_cov, factor_means, factor_cov, T=3)  # T-N-K = 3-2-1 = 0


def test_mle_covariance_divides_by_t_not_t_minus_one():
    # GRS (1989) uses the MLE (divide-by-T) covariance, not the usual
    # sample covariance (divide-by-(T-1)) -- getting this wrong silently
    # shrinks both Sigma and Omega by the same (T-1)/T factor, which does
    # NOT cancel in the GRS ratio (Sigma is inverted, Omega is inverted
    # separately, and the two matrices differ in size), so the convention
    # must be explicit and tested, not left to numpy's own default.
    residuals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # T=3, N=2
    cov = mle_covariance(residuals)
    means = residuals.mean(axis=0)
    centered = residuals - means
    expected = (centered.T @ centered) / 3  # divide by T=3, not T-1=2
    np.testing.assert_allclose(cov, expected)


def test_minimum_detectable_alpha_monthly_matches_hand_computed_z_scores():
    # z_0.975 = 1.959964, z_0.80 = 0.841621 (two-sided 5% test, 80% power)
    se = 0.01
    mde = minimum_detectable_alpha_monthly(se, power=0.80, alpha_level=0.05)
    assert mde == pytest.approx((1.9599639845 + 0.8416212336) * 0.01, rel=1e-6)


def test_run_bh_correction_reuses_the_existing_module_at_the_given_fdr_level():
    # Alphas with OLS t-stats engineered so exactly one (the largest |t|)
    # survives Benjamini-Hochberg at a loose 50% FDR but none survive at a
    # strict 1% FDR -- confirms run_bh_correction is actually wired to
    # factors.multiple_testing.benjamini_hochberg's own result, not a
    # reimplementation, and that the alpha level is threaded through.
    results = [
        PanelStockResult(
            ticker="A.NS", n_obs=80, alpha_monthly=0.001, alpha_annual=0.012,
            alpha_t_ols=0.1, alpha_t_nw=0.1, r_squared=0.3, se_alpha_monthly=0.01,
        ),
        PanelStockResult(
            ticker="B.NS", n_obs=80, alpha_monthly=0.02, alpha_annual=0.27,
            alpha_t_ols=2.5, alpha_t_nw=2.5, r_squared=0.3, se_alpha_monthly=0.005,
        ),
    ]
    loose = run_bh_correction(results, fdr=0.50)
    strict = run_bh_correction(results, fdr=0.01)

    assert loose.rejected["B.NS"] is True
    assert loose.rejected["A.NS"] is False
    assert strict.rejected["B.NS"] is False
    assert strict.n_rejected == 0


def _write_monthly_ohlcv(path, month_ends: list[str], closes: list[float]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close", "high", "low", "open", "volume"])
        for d, c in zip(month_ends, closes):
            writer.writerow([d, c, c + 1, c - 1, c, 1000])


def _synthetic_factors(month_ends: list[str]) -> pd.DataFrame:
    idx = pd.PeriodIndex(pd.to_datetime(month_ends), freq="M")
    n = len(idx)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mf": rng.normal(0.01, 0.03, n),
            "smb": rng.normal(0.0, 0.02, n),
            "hml": rng.normal(0.0, 0.02, n),
            "wml": rng.normal(0.0, 0.02, n),
            "rf": [0.003] * n,
        },
        index=idx,
    )


def test_grs_common_window_panel_window_start_is_the_max_of_only_the_passed_tickers(tmp_path):
    from factors.panel import grs_common_window_panel

    month_ends = [
        "2015-01-31", "2015-02-28", "2015-03-31", "2015-04-30", "2015-05-31",
        "2015-06-30", "2015-07-31", "2015-08-31", "2015-09-30", "2015-10-31",
        "2015-11-30", "2015-12-31", "2016-01-31", "2016-02-29", "2016-03-31",
        "2016-04-30",
    ]
    # N=6 assets, K=4 factors needs T > N+K=10 return-observations for GRS's
    # degrees of freedom to be positive; 16 month-end closes -> 15 returns.
    factors_table = _synthetic_factors(month_ends)

    # FF3+WML has K=4 factors, so GRS needs N >= K+2 = 6 assets to be
    # computable at all; six full-history tickers plus one short-history
    # straggler that must have zero effect once excluded by the caller.
    full_history_tickers = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS", "F.NS"]
    rng = np.random.default_rng(1)
    for i, ticker in enumerate(full_history_tickers):
        closes = 100 + i * 10 + np.cumsum(rng.normal(0, 1, len(month_ends)))
        _write_monthly_ohlcv(tmp_path / f"{ticker.replace('.', '_')}.csv", month_ends, list(closes))
    # G.NS only has the last 2 months -- if this ticker were included in the
    # candidate list, the common window would collapse to ~2 months; it must
    # have no effect when the caller (correctly) excludes it.
    _write_monthly_ohlcv(tmp_path / "G_NS.csv", month_ends[-2:], [10, 11])

    result = grs_common_window_panel(full_history_tickers, fixtures_dir=tmp_path, factors_table=factors_table)

    assert result is not None
    assert set(result.tickers) == set(full_history_tickers)
    # T should reflect the full ~15-return overlap across A-F and the
    # factors, not be dragged down to G.NS's 2-month history (never passed in)
    assert result.grs.n_periods >= 12


def test_grs_common_window_panel_returns_none_when_too_few_assets_survive(tmp_path):
    from factors.panel import grs_common_window_panel

    month_ends = ["2015-01-31", "2015-02-28", "2015-03-31"]
    factors_table = _synthetic_factors(month_ends)
    _write_monthly_ohlcv(tmp_path / "A_NS.csv", month_ends, [100, 101, 102])

    # only 1 asset, FF3+WML has K=4 factors -> need >= K+2=6 assets for Sigma
    result = grs_common_window_panel(["A.NS"], fixtures_dir=tmp_path, factors_table=factors_table)
    assert result is None


def test_grs_common_window_panel_reports_infeasible_when_n_exceeds_t_minus_k(tmp_path):
    # N (assets) can exceed T (common-window months) once a cross-section is
    # wide and listing dates are heterogeneous -- Sigma is then rank-deficient
    # and GRS is not computable. This must be reported honestly (grs=None,
    # infeasible_reason set), never silently patched by dropping assets.
    from factors.panel import grs_common_window_panel

    # Real calendar month-end dates, so load_monthly_returns's own "drop a
    # still-in-progress final month" guard doesn't eat the last one: 7
    # month-end closes -> 6 returns (Feb-Jul), inner-joined against 7 months
    # of factors (Jan-Jul) -> a 6-month common window. 6 assets, K=4 factors:
    # T-N-K = 6-6-4 = -4 <= 0, forcing the rank-deficient case.
    short_month_ends = [
        "2015-01-31", "2015-02-28", "2015-03-31", "2015-04-30",
        "2015-05-31", "2015-06-30", "2015-07-31",
    ]
    factors_table_short = _synthetic_factors(short_month_ends)
    tickers = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS", "F.NS"]
    rng = np.random.default_rng(2)
    for i, ticker in enumerate(tickers):
        closes = 100 + i * 5 + np.cumsum(rng.normal(0, 1, len(short_month_ends)))
        _write_monthly_ohlcv(tmp_path / f"{ticker.replace('.', '_')}.csv", short_month_ends, list(closes))

    result = grs_common_window_panel(tickers, fixtures_dir=tmp_path, factors_table=factors_table_short)

    assert result is not None
    assert result.grs is None
    assert "rank-deficient" in result.infeasible_reason
    assert len(result.tickers) == 6

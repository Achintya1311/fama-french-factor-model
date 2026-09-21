import numpy as np
import pandas as pd
import pytest

from factors.diagnostics import (
    confidence_band,
    newey_west_lag,
    rolling_beta,
    rolling_loadings,
    run_diagnostics,
)


def _synthetic_factors(n=240, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2000-01", periods=n, freq="M")
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.005, 0.04, n),
            "rf": 0.001,
        },
        index=index,
    )


def test_newey_west_lag_matches_known_values():
    # Newey & West (1994): floor(4*(n/100)**(2/9)). Hand-checked reference
    # points, not just re-deriving the same formula the code uses.
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1200) == 6
    assert newey_west_lag(1) == 1  # floored at 1, never 0


def test_iid_residuals_pass_both_diagnostic_tests():
    # No autocorrelation and no heteroskedasticity built in, by construction:
    # residuals should not be flagged by either test, and the Newey-West
    # t-stat should land close to the plain-OLS one (nothing for HAC to
    # correct).
    factors = _synthetic_factors(n=600, seed=1)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.02, len(factors))
    portfolio = 0.001 + 1.0 * factors["mkt_rf"] + factors["rf"] + noise

    result = run_diagnostics(portfolio, factors, ("mkt_rf",))

    assert not result.autocorrelated
    assert not result.heteroskedastic
    assert result.alpha_t_nw == pytest.approx(result.alpha_t_ols, rel=0.5)
    assert 1.5 < result.durbin_watson < 2.5


def test_autocorrelated_residuals_are_flagged_and_move_the_tstat():
    # Build an AR(1) residual series with a strong positive coefficient:
    # Ljung-Box must flag it, and Newey-West's correction must actually
    # change the alpha t-stat relative to plain OLS (that's the entire
    # point of adding it).
    factors = _synthetic_factors(n=600, seed=3)
    rng = np.random.default_rng(4)
    innovations = rng.normal(0, 0.02, len(factors))
    ar_resid = np.zeros(len(factors))
    for i in range(1, len(ar_resid)):
        ar_resid[i] = 0.85 * ar_resid[i - 1] + innovations[i]
    portfolio = 0.001 + 1.0 * factors["mkt_rf"] + factors["rf"] + ar_resid

    result = run_diagnostics(portfolio, factors, ("mkt_rf",))

    assert result.autocorrelated
    assert result.durbin_watson < 1.5
    assert result.alpha_t_nw != pytest.approx(result.alpha_t_ols, rel=0.05)


def test_heteroskedastic_residuals_are_flagged():
    # Residual variance scales with the factor itself (a classic funnel
    # shape): Breusch-Pagan should catch it.
    factors = _synthetic_factors(n=600, seed=5)
    rng = np.random.default_rng(6)
    scale = 0.005 + 0.3 * factors["mkt_rf"].abs()
    noise = rng.normal(0, 1, len(factors)) * scale
    portfolio = 0.001 + 1.0 * factors["mkt_rf"] + factors["rf"] + noise

    result = run_diagnostics(portfolio, factors, ("mkt_rf",))

    assert result.heteroskedastic


def test_run_diagnostics_reports_requested_factor_columns_and_lag():
    factors = _synthetic_factors(n=300, seed=7)
    portfolio = 0.001 + factors["mkt_rf"] + factors["rf"]
    result = run_diagnostics(portfolio, factors, ("mkt_rf",), ljung_box_lag=6)
    assert result.factor_columns == ("mkt_rf",)
    assert result.ljung_box_lag == 6
    assert result.nw_lag == newey_west_lag(result.n_obs)
    assert set(result.loadings_t_nw) == {"mkt_rf"}


def test_rolling_beta_recovers_a_regime_shift():
    # First half beta=0.5, second half beta=1.5, no noise: a 24-month
    # rolling window should read close to 0.5 well inside the first regime
    # and close to 1.5 well inside the second, with the crossover in
    # between - not an exact date match, just "it moved the right way".
    n = 120
    index = pd.period_range("2000-01", periods=n, freq="M")
    rng = np.random.default_rng(8)
    mkt_rf = pd.Series(rng.normal(0.005, 0.04, n), index=index)
    factors = pd.DataFrame({"mkt_rf": mkt_rf, "rf": 0.001}, index=index)
    beta = pd.Series([0.5] * (n // 2) + [1.5] * (n // 2), index=index)
    portfolio = beta * factors["mkt_rf"] + factors["rf"]

    betas = rolling_beta(portfolio, factors, window=24)

    assert len(betas) == n - 24 + 1
    assert betas.iloc[10] == pytest.approx(0.5, abs=1e-6)
    assert betas.iloc[-1] == pytest.approx(1.5, abs=1e-6)


def test_rolling_beta_requires_a_full_window():
    factors = _synthetic_factors(n=10, seed=9)
    portfolio = factors["mkt_rf"] + factors["rf"]
    betas = rolling_beta(portfolio, factors, window=24)
    assert betas.empty


def _synthetic_multifactor(n=240, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2000-01", periods=n, freq="M")
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.005, 0.04, n),
            "smb": rng.normal(0.0, 0.03, n),
            "hml": rng.normal(0.0, 0.03, n),
            "rf": 0.001,
        },
        index=index,
    )


def test_rolling_loadings_recovers_a_regime_shift_across_factors():
    # mkt_rf loading 0.5 -> 1.5 and smb loading 1.0 -> 0.0 at the halfway
    # point, no noise: a 24-month rolling window should read close to each
    # regime's own loadings well inside it, for every requested factor at
    # once (rolling_beta only ever proved this for a single factor).
    n = 120
    factors = _synthetic_multifactor(n=n, seed=10)
    mkt_beta = pd.Series([0.5] * (n // 2) + [1.5] * (n // 2), index=factors.index)
    smb_beta = pd.Series([1.0] * (n // 2) + [0.0] * (n // 2), index=factors.index)
    portfolio = mkt_beta * factors["mkt_rf"] + smb_beta * factors["smb"] + factors["rf"]

    result = rolling_loadings(portfolio, factors, ("mkt_rf", "smb"), window=24)

    assert result.factor_columns == ("mkt_rf", "smb")
    assert len(result.params) == n - 24 + 1
    assert result.params["mkt_rf"].iloc[10] == pytest.approx(0.5, abs=1e-6)
    assert result.params["mkt_rf"].iloc[-1] == pytest.approx(1.5, abs=1e-6)
    assert result.params["smb"].iloc[10] == pytest.approx(1.0, abs=1e-6)
    assert result.params["smb"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    # Zero noise means an exact fit and (numerically) a zero standard error,
    # for any window sitting entirely inside one regime - a window straddling
    # the regime change is fitting two different true loadings at once, so
    # it's expected to have nonzero residual/SE and is deliberately not
    # checked here.
    assert result.se["mkt_rf"].iloc[10] == pytest.approx(0.0, abs=1e-6)
    assert result.se["mkt_rf"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert result.se["smb"].iloc[10] == pytest.approx(0.0, abs=1e-6)
    assert result.se["smb"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert list(result.se.columns) == ["mkt_rf", "smb"]
    assert result.se.index.equals(result.params.index)


def test_rolling_loadings_requires_a_full_window():
    factors = _synthetic_multifactor(n=10, seed=11)
    portfolio = factors["mkt_rf"] + factors["rf"]
    result = rolling_loadings(portfolio, factors, ("mkt_rf", "smb"), window=24)
    assert result.params.empty
    assert result.se.empty
    assert list(result.params.columns) == ["mkt_rf", "smb"]


def test_confidence_band_brackets_the_point_estimate_and_widens_with_z():
    factors = _synthetic_multifactor(n=300, seed=12)
    rng = np.random.default_rng(13)
    noise = rng.normal(0, 0.02, len(factors))
    portfolio = 0.001 + 1.0 * factors["mkt_rf"] + 0.3 * factors["smb"] + factors["rf"] + noise

    result = rolling_loadings(portfolio, factors, ("mkt_rf", "smb"), window=60)
    lower, upper = confidence_band(result.params, result.se)

    assert (lower <= result.params).all().all()
    assert (result.params <= upper).all().all()

    wide_lower, wide_upper = confidence_band(result.params, result.se, z=3.0)
    assert (wide_lower <= lower).all().all()
    assert (wide_upper >= upper).all().all()

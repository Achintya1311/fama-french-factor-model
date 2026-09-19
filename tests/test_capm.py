import numpy as np
import pandas as pd
import pytest

from factors.capm import align_excess_returns, run_capm
from factors.kenfrench import load_monthly_factors


def _synthetic_factors(n=240, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2000-01", periods=n, freq="M")
    mkt_rf = pd.Series(rng.normal(0.005, 0.04, n), index=index)
    rf = pd.Series(0.001, index=index)
    return pd.DataFrame({"mkt_rf": mkt_rf, "smb": 0.0, "hml": 0.0, "rf": rf})


def test_recovers_known_alpha_and_beta_from_synthetic_data():
    # Build a portfolio with a *known* alpha/beta and no noise: recovering
    # them exactly is what "the regression machinery is correct" means,
    # independent of anything about real markets.
    factors = _synthetic_factors()
    true_alpha, true_beta = 0.003, 1.4
    excess = true_alpha + true_beta * factors["mkt_rf"]
    portfolio = excess + factors["rf"]

    result = run_capm(portfolio, factors)

    assert result.alpha_monthly == pytest.approx(true_alpha, abs=1e-10)
    assert result.beta == pytest.approx(true_beta, abs=1e-10)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.n_obs == len(factors)


def test_alpha_annualizes_by_compounding():
    factors = _synthetic_factors()
    alpha, beta = 0.01, 1.0
    portfolio = alpha + beta * factors["mkt_rf"] + factors["rf"]
    result = run_capm(portfolio, factors)
    assert result.alpha_annual == pytest.approx((1 + alpha) ** 12 - 1, abs=1e-9)


def test_significant_flag_matches_5pct_two_sided_threshold():
    factors = _synthetic_factors(n=600, seed=1)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.02, len(factors))
    # Tiny true alpha plus noise: not a fixed-point test of the exact
    # t-stat (that depends on the noise draw), just that the flag agrees
    # with the |t| > 1.96 rule the dataclass documents.
    portfolio = 0.0001 + 1.0 * factors["mkt_rf"] + factors["rf"] + noise
    result = run_capm(portfolio, factors)
    assert result.significant == (abs(result.alpha_t) > 1.96)


def test_align_excess_returns_requires_overlap():
    factors = _synthetic_factors()
    disjoint = pd.Series(0.01, index=pd.period_range("1900-01", periods=12, freq="M"))
    with pytest.raises(ValueError, match="no overlapping months"):
        align_excess_returns(disjoint, factors)


def test_align_excess_returns_subtracts_rf():
    factors = _synthetic_factors(n=5)
    portfolio = factors["rf"] + 0.02
    aligned = align_excess_returns(portfolio, factors)
    assert np.allclose(aligned["excess"], 0.02)


def test_market_regressed_on_itself_is_the_known_answer():
    # Using the real committed fixture: reconstructing the market portfolio
    # as mkt_rf + rf and regressing it on mkt_rf must return alpha=0,
    # beta=1, R-squared=1 - an exact analytical identity, not an estimate.
    # This is the machinery-correctness check for Day 2; replicating a
    # *published* FF3 result on an independent portfolio is Day 3's job.
    factors = load_monthly_factors()
    market_proxy = factors["mkt_rf"] + factors["rf"]
    result = run_capm(market_proxy, factors)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    assert result.beta == pytest.approx(1.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)

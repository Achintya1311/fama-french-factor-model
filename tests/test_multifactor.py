import numpy as np
import pandas as pd
import pytest

from factors.kenfrench import load_monthly_factors
from factors.multifactor import (
    FF3_COLUMNS,
    FF5_MOM_COLUMNS,
    align_excess_returns,
    load_ff5_mom_factors,
    run_factor_model,
)


def _synthetic_ff3_factors(n=240, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2000-01", periods=n, freq="M")
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.005, 0.04, n),
            "smb": rng.normal(0.001, 0.02, n),
            "hml": rng.normal(0.001, 0.02, n),
            "rf": 0.001,
        },
        index=index,
    )


def test_recovers_known_alpha_and_loadings_from_synthetic_data():
    # Same idea as capm.py's synthetic test, extended to three factors: a
    # noiseless portfolio with known alpha/loadings must come back exact -
    # that's what "the multi-factor OLS wiring is correct" means.
    factors = _synthetic_ff3_factors()
    true_alpha = 0.003
    true_loadings = {"mkt_rf": 1.2, "smb": 0.5, "hml": -0.3}
    excess = true_alpha + sum(true_loadings[c] * factors[c] for c in FF3_COLUMNS)
    portfolio = excess + factors["rf"]

    result = run_factor_model(portfolio, factors, FF3_COLUMNS)

    assert result.alpha_monthly == pytest.approx(true_alpha, abs=1e-10)
    for c in FF3_COLUMNS:
        assert result.loadings[c] == pytest.approx(true_loadings[c], abs=1e-10)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.n_obs == len(factors)


def test_align_excess_returns_requires_overlap():
    factors = _synthetic_ff3_factors()
    disjoint = pd.Series(0.01, index=pd.period_range("1900-01", periods=12, freq="M"))
    with pytest.raises(ValueError, match="no overlapping months"):
        align_excess_returns(disjoint, factors, FF3_COLUMNS)


def test_align_excess_returns_requires_factor_columns():
    factors = _synthetic_ff3_factors()
    portfolio = factors["rf"]
    with pytest.raises(ValueError, match="missing columns"):
        align_excess_returns(portfolio, factors, ("rmw",))


# --- Day 3's "replicates a published/known FF3 result" gate ---
#
# There's no internet access in this sandbox to an actual journal figure to
# replicate against, so the machinery-correctness check Day 2 used for CAPM
# (market regressed on itself is an exact analytical identity) is extended
# here to three and six factors: each Fama-French factor is itself a
# published, independent long/short portfolio return series, and regressing
# it on the full factor set it belongs to has a known-exact answer (alpha=0,
# its own loading=1, every other loading=0, R2=1) - not an estimate, an
# identity. A bug in the multi-factor wiring (wrong column order, a
# transposed design matrix, RF subtracted twice) fails this at machine
# precision; a correct implementation cannot.


def test_market_on_ff3_is_the_known_identity():
    factors = load_monthly_factors()
    market_proxy = factors["mkt_rf"] + factors["rf"]
    result = run_factor_model(market_proxy, factors, FF3_COLUMNS)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["mkt_rf"] == pytest.approx(1.0, abs=1e-9)
    assert result.loadings["smb"] == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["hml"] == pytest.approx(0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_smb_on_ff3_is_the_known_identity():
    factors = load_monthly_factors()
    smb_proxy = factors["smb"] + factors["rf"]
    result = run_factor_model(smb_proxy, factors, FF3_COLUMNS)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["mkt_rf"] == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["smb"] == pytest.approx(1.0, abs=1e-9)
    assert result.loadings["hml"] == pytest.approx(0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_hml_on_ff3_is_the_known_identity():
    factors = load_monthly_factors()
    hml_proxy = factors["hml"] + factors["rf"]
    result = run_factor_model(hml_proxy, factors, FF3_COLUMNS)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["mkt_rf"] == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["smb"] == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["hml"] == pytest.approx(1.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_load_ff5_mom_factors_has_all_six_columns():
    factors = load_ff5_mom_factors()
    assert list(factors.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]
    assert isinstance(factors.index, pd.PeriodIndex)
    assert factors.index.is_unique
    assert not factors.empty


def test_rmw_on_ff5_mom_is_the_known_identity():
    factors = load_ff5_mom_factors()
    rmw_proxy = factors["rmw"] + factors["rf"]
    result = run_factor_model(rmw_proxy, factors, FF5_MOM_COLUMNS)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    for c in FF5_MOM_COLUMNS:
        assert result.loadings[c] == pytest.approx(1.0 if c == "rmw" else 0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_mom_on_ff5_mom_is_the_known_identity():
    factors = load_ff5_mom_factors()
    mom_proxy = factors["mom"] + factors["rf"]
    result = run_factor_model(mom_proxy, factors, FF5_MOM_COLUMNS)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    for c in FF5_MOM_COLUMNS:
        assert result.loadings[c] == pytest.approx(1.0 if c == "mom" else 0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)

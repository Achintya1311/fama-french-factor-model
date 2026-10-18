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


# --- D15: a regression fit near its own parameter count is rank-deficient
# or exactly-determined, not a real finding (critique
# projects/portfolio-fixes/critiques/2026-10-18-fama-french-factor-model-1.md) ---


@pytest.mark.parametrize("n", [4, 5, 6])
def test_run_factor_model_rejects_degenerate_few_month_samples(n):
    # FF3 has 3 factor columns + intercept = 4 parameters; 4/5/6 months
    # reproduce the critique's exact rank-deficient/exactly-determined cases
    # (a SingularMatrixWarning-triggering design matrix, not a genuine null
    # result) rather than letting them silently return a fabricated
    # R-squared=1.0 "result".
    factors = _synthetic_ff3_factors(n=n, seed=0)
    portfolio = factors["mkt_rf"] + factors["rf"]
    with pytest.raises(ValueError, match=f"n_obs={n}"):
        run_factor_model(portfolio, factors, FF3_COLUMNS)


def test_run_factor_model_enforces_the_3x_parameter_count_floor():
    # FF3 has 4 parameters (3 factor columns + intercept); the floor is
    # 3x that, i.e. n_obs >= 12. 11 must fail, 12 must pass.
    factors = _synthetic_ff3_factors(n=11, seed=1)
    portfolio = factors["mkt_rf"] + factors["rf"]
    with pytest.raises(ValueError, match="n_obs=11"):
        run_factor_model(portfolio, factors, FF3_COLUMNS)

    factors_ok = _synthetic_ff3_factors(n=12, seed=1)
    portfolio_ok = factors_ok["mkt_rf"] + factors_ok["rf"]
    result = run_factor_model(portfolio_ok, factors_ok, FF3_COLUMNS)
    assert result.n_obs == 12


def test_run_factor_model_error_names_n_obs_and_factor_count():
    factors = _synthetic_ff3_factors(n=6, seed=2)
    portfolio = factors["mkt_rf"] + factors["rf"]
    with pytest.raises(ValueError, match=r"n_obs=6 against 3 factor columns"):
        run_factor_model(portfolio, factors, FF3_COLUMNS)


def test_run_factor_model_22_month_ff5mom_screen_check_case_is_unaffected():
    # The real-world floor-margin case this fix must not break: Day 5-7's
    # 22-month Ken French screen check regresses FF5+Mom's 6 factor columns
    # (7 parameters incl. intercept), needing n_obs >= 21 under the new
    # floor -- 22 passes, by a 1-month margin.
    from factors.multifactor import FF5_MOM_COLUMNS

    rng = np.random.default_rng(3)
    index = pd.period_range("2000-01", periods=22, freq="M")
    factors = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.005, 0.04, 22),
            "smb": rng.normal(0.001, 0.02, 22),
            "hml": rng.normal(0.001, 0.02, 22),
            "rmw": rng.normal(0.001, 0.02, 22),
            "cma": rng.normal(0.001, 0.02, 22),
            "mom": rng.normal(0.001, 0.02, 22),
            "rf": 0.001,
        },
        index=index,
    )
    portfolio = factors["mkt_rf"] + factors["rf"]
    result = run_factor_model(portfolio, factors, FF5_MOM_COLUMNS)
    assert result.n_obs == 22


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

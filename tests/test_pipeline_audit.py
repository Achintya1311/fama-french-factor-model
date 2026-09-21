import numpy as np
import pandas as pd
import pytest

from factors.diagnostics import rolling_beta
from factors.pipeline_audit import (
    alignment_is_order_invariant,
    rolling_beta_causality_violations,
    rolling_loadings_causality_violations,
    rolling_series_causality_violations,
    run_alignment_audit,
    run_multiple_testing_audit,
)


def _synthetic_factors(n=240, seed=0):
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


def _regime_shift_portfolio(factors, n, seed=1):
    rng = np.random.default_rng(seed)
    beta = pd.Series([0.5] * (n // 2) + [1.5] * (n // 2), index=factors.index)
    noise = rng.normal(0, 0.005, n)
    return beta * factors["mkt_rf"] + factors["rf"] + noise


# --- causality: positive control (the repo's real, trailing rolling_beta) --


def test_rolling_beta_is_causal_on_real_regime_shift_data():
    n = 240
    factors = _synthetic_factors(n=n, seed=0)
    portfolio = _regime_shift_portfolio(factors, n)

    violations = rolling_beta_causality_violations(portfolio, factors, "mkt_rf", window=24)

    assert violations == []


def test_rolling_loadings_is_causal_on_real_regime_shift_data():
    n = 240
    factors = _synthetic_factors(n=n, seed=2)
    rng = np.random.default_rng(3)
    mkt_beta = pd.Series([0.5] * (n // 2) + [1.5] * (n // 2), index=factors.index)
    smb_beta = pd.Series([1.0] * (n // 2) + [0.0] * (n // 2), index=factors.index)
    noise = rng.normal(0, 0.005, n)
    portfolio = mkt_beta * factors["mkt_rf"] + smb_beta * factors["smb"] + factors["rf"] + noise

    violations = rolling_loadings_causality_violations(portfolio, factors, ("mkt_rf", "smb"), window=24)

    assert violations == []


# --- causality: negative control (a deliberately non-causal series) -------


def _lookahead_rolling_beta(portfolio, factors, factor_column="mkt_rf", window=24, peek=6):
    """A synthetic 'rolling beta' that reports month i's value using data
    through month i+peek - a manufactured look-ahead bug, used only to prove
    the checker actually fires on broken input rather than always passing.
    """
    good = rolling_beta(portfolio, factors, factor_column, window)
    return good.shift(-peek).dropna()


def test_causality_check_detects_a_manufactured_lookahead_violation():
    n = 240
    factors = _synthetic_factors(n=n, seed=4)
    portfolio = _regime_shift_portfolio(factors, n, seed=5)

    def bad_fn(p, f):
        return _lookahead_rolling_beta(p, f, "mkt_rf", window=24, peek=6)

    violations = rolling_series_causality_violations("bad", bad_fn, portfolio, factors, stride=24)

    assert violations != []


# --- alignment order-invariance: positive control (the repo's real joins) -


def test_capm_alignment_is_order_invariant_on_real_data():
    from factors.capm import align_excess_returns as capm_align

    n = 60
    factors = _synthetic_factors(n=n, seed=6)
    portfolio = _regime_shift_portfolio(factors, n, seed=7)

    assert alignment_is_order_invariant(capm_align, portfolio, factors) is True


def test_multifactor_alignment_is_order_invariant_on_real_data():
    from factors.multifactor import FF3_COLUMNS, align_excess_returns as multifactor_align

    n = 60
    factors = _synthetic_factors(n=n, seed=8)
    portfolio = _regime_shift_portfolio(factors, n, seed=9)

    def align_fn(p, f):
        return multifactor_align(p, f, FF3_COLUMNS)

    assert alignment_is_order_invariant(align_fn, portfolio, factors) is True


# --- alignment order-invariance: negative control (a positional join) -----


def _bad_positional_align(portfolio, factors):
    """Joins by row position instead of month label - exactly the class of
    bug ``alignment_is_order_invariant`` exists to catch.
    """
    aligned = pd.DataFrame(
        {
            "excess": portfolio.reset_index(drop=True) - factors["rf"].reset_index(drop=True),
            "mkt_rf": factors["mkt_rf"].reset_index(drop=True),
        }
    )
    aligned.index = portfolio.index[: len(aligned)]
    return aligned


def test_alignment_check_detects_a_manufactured_positional_join():
    n = 60
    factors = _synthetic_factors(n=n, seed=10)
    portfolio = _regime_shift_portfolio(factors, n, seed=11)

    assert alignment_is_order_invariant(_bad_positional_align, portfolio, factors) is False


# --- integration: the real audit runs clean on the repo's own fixtures ----


def test_run_alignment_audit_is_clean_on_committed_fixtures():
    report = run_alignment_audit(window=60)

    assert report["causality_checked"]
    assert report["causality_violations"] == {}
    assert report["order_checked"]
    assert report["order_violations"] == []


def test_run_multiple_testing_audit_reports_all_four_families():
    report = run_multiple_testing_audit(alpha=0.05)

    assert set(report) == {
        "decay_report (OLS)",
        "decay_report (Newey-West)",
        "screen_check (OLS)",
        "screen_check (Newey-West)",
    }
    assert report["decay_report (OLS)"]["n_tests"] == 30
    for name, result in report.items():
        # Bonferroni is the stricter (family-wise) correction: it can never
        # let more survive than the raw, uncorrected count. BH sits between.
        assert len(result["bonferroni_survivors"]) <= len(result["raw_significant"])
        assert set(result["bonferroni_survivors"]).issubset(result["bh_survivors"])
        assert set(result["bh_survivors"]).issubset(result["raw_significant"])

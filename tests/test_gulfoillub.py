from pathlib import Path

import pandas as pd
import pytest

from factors.gulfoillub import (
    CONTRACT_MODEL,
    DEFAULT_OHLCV_PATH,
    MODEL_FACTOR_COLUMNS,
    evaluate,
    load_gulfoillub_returns,
    main,
    to_contract,
)


def test_gulfoillub_fixture_is_committed():
    assert Path(DEFAULT_OHLCV_PATH).exists()


def test_load_gulfoillub_returns_is_period_indexed_monthly():
    returns = load_gulfoillub_returns()
    assert isinstance(returns.index, pd.PeriodIndex)
    assert returns.index.freqstr == "M"
    assert returns.index.is_monotonic_increasing


@pytest.mark.parametrize("model", ["capm", "ff3", "ff3wml"])
def test_evaluate_overlaps_137_months_with_iima(model):
    # GULFOILLUB's OHLCV fixture runs past 2025-12, but IIMA's factor
    # history stops there, so the inner join caps every spec at the same
    # 137 months (2014-08 to 2025-12) regardless of which columns are used.
    result, _ = evaluate(model)
    assert result.n_obs == 137
    assert result.factor_columns == MODEL_FACTOR_COLUMNS[model]


def test_capm_spec_reconstructs_the_market_exactly():
    # Same exact-identity check Day 2/3 ran for Ken French's own factors:
    # regressing (mf + rf) - the market portfolio itself - on mf must
    # return alpha=0, loading=1, R2=1, or the OLS wiring has a bug.
    idx = pd.period_range("2020-01", periods=24, freq="M")
    factors_table = pd.DataFrame(
        {
            "mf": [0.01 * ((i % 5) - 2) for i in range(24)],
            "smb": [0.002 * i for i in range(24)],
            "hml": [-0.001 * i for i in range(24)],
            "wml": [0.0015 * ((i % 3) - 1) for i in range(24)],
            "rf": [0.003] * 24,
        },
        index=idx,
    )
    market = factors_table["mf"] + factors_table["rf"]
    result, _ = evaluate("capm", returns=market, factors_table=factors_table)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["mf"] == pytest.approx(1.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_ff3wml_spec_recovers_known_synthetic_loadings():
    idx = pd.period_range("2020-01", periods=30, freq="M")
    factors_table = pd.DataFrame(
        {
            "mf": [0.01 * ((i % 5) - 2) for i in range(30)],
            "smb": [0.004 * ((i % 4) - 1.5) for i in range(30)],
            "hml": [-0.003 * ((i % 3) - 1) for i in range(30)],
            "wml": [0.002 * ((i % 6) - 2.5) for i in range(30)],
            "rf": [0.003] * 30,
        },
        index=idx,
    )
    # Build a portfolio with known loadings and zero alpha/noise: excess
    # return is an exact linear combination of the four factors.
    excess = 1.2 * factors_table["mf"] - 0.5 * factors_table["smb"] + 0.3 * factors_table["hml"] + 0.8 * factors_table["wml"]
    portfolio = excess + factors_table["rf"]
    result, _ = evaluate("ff3wml", returns=portfolio, factors_table=factors_table)
    assert result.alpha_monthly == pytest.approx(0.0, abs=1e-9)
    assert result.loadings["mf"] == pytest.approx(1.2, abs=1e-9)
    assert result.loadings["smb"] == pytest.approx(-0.5, abs=1e-9)
    assert result.loadings["hml"] == pytest.approx(0.3, abs=1e-9)
    assert result.loadings["wml"] == pytest.approx(0.8, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_to_contract_has_the_v0_2_shape_with_mkt_renamed_from_mf():
    result, _ = evaluate(CONTRACT_MODEL)
    contract = to_contract(result)
    factor = contract["factor"]
    assert set(factor) == {"alpha_annual", "alpha_t", "significant", "loadings"}
    assert isinstance(factor["alpha_annual"], float)
    assert isinstance(factor["alpha_t"], float)
    assert isinstance(factor["significant"], bool)
    assert "mkt" in factor["loadings"]
    assert "mf" not in factor["loadings"]


def test_to_contract_carries_the_extra_wml_loading():
    # gates.factor.load_contract (STOCKSTALKER) only checks the four
    # top-level keys, not which keys `loadings` itself has, so this extra
    # key - no Ken French contract has ever shipped one - passes through.
    result, _ = evaluate("ff3wml")
    contract = to_contract(result)
    assert set(contract["factor"]["loadings"]) == {"mkt", "smb", "hml", "wml"}


def test_contract_restricted_to_the_shipped_model():
    with pytest.raises(SystemExit):
        main(["--model", "capm", "--contract", "/tmp/should-not-be-written.json"])
    with pytest.raises(SystemExit):
        main(["--model", "ff3", "--contract", "/tmp/should-not-be-written.json"])


def test_cli_writes_contract_file(tmp_path):
    out = tmp_path / "GULFOILLUB.json"
    main(["--model", "ff3wml", "--contract", str(out)])
    import json

    data = json.loads(out.read_text())
    assert set(data["factor"]) == {"alpha_annual", "alpha_t", "significant", "loadings"}


def test_cli_diagnostics_flag_runs_without_error(capsys):
    main(["--model", "ff3wml", "--diagnostics"])
    captured = capsys.readouterr()
    assert "alpha t-stat (NW)" in captured.out


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        evaluate("ff5mom")


def test_cli_converts_insufficient_data_to_a_one_line_system_exit(monkeypatch, capsys):
    # D15: main() must print the insufficient-data reason as a clean
    # one-liner (SystemExit, no traceback) and never reach the --contract
    # write -- the GULFOILLUB fixture itself is always long enough (137
    # months), so this exercises the CLI wiring directly by making
    # evaluate() raise the way a thinner ticker's regression would.
    import factors.gulfoillub as gulfoillub_module

    def _raise(model, returns=None, factors_table=None):
        raise ValueError("too few observations for a factor regression: n_obs=6 against 3 factor columns")

    monkeypatch.setattr(gulfoillub_module, "evaluate", _raise)
    with pytest.raises(SystemExit, match="n_obs=6"):
        main(["--model", "ff3wml"])
    assert capsys.readouterr().out == ""

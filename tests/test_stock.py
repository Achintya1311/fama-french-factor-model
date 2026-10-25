from pathlib import Path

import pandas as pd
import pytest

from factors import gulfoillub
from factors.stock import (
    CONTRACT_MODEL,
    MODEL_FACTOR_COLUMNS,
    evaluate,
    load_stock_returns,
    main,
    to_contract,
)

GULFOILLUB_PRICES = gulfoillub.DEFAULT_OHLCV_PATH


def test_load_stock_returns_is_period_indexed_monthly():
    returns = load_stock_returns(GULFOILLUB_PRICES)
    assert isinstance(returns.index, pd.PeriodIndex)
    assert returns.index.freqstr == "M"
    assert returns.index.is_monotonic_increasing


@pytest.mark.parametrize("model", ["capm", "ff3", "ff3wml"])
def test_evaluate_overlaps_137_months_with_iima_on_gulfoillubs_own_fixture(model):
    result, _ = evaluate(model, prices_csv=GULFOILLUB_PRICES)
    assert result.n_obs == 137
    assert result.factor_columns == MODEL_FACTOR_COLUMNS[model]


def test_evaluate_requires_either_prices_csv_or_returns():
    with pytest.raises(ValueError, match="needs either prices_csv or returns"):
        evaluate("ff3wml")


def test_capm_spec_reconstructs_the_market_exactly():
    # Same exact-identity check gulfoillub.py's own test runs: regressing
    # (mf + rf) -- the market portfolio itself -- on mf must return
    # alpha=0, loading=1, R2=1, or the OLS wiring has a bug.
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


def test_to_contract_has_the_v0_2_shape_with_mkt_renamed_from_mf():
    result, _ = evaluate(CONTRACT_MODEL, prices_csv=GULFOILLUB_PRICES)
    contract = to_contract(result)
    factor = contract["factor"]
    assert set(factor) == {"alpha_annual", "alpha_t", "significant", "loadings"}
    assert isinstance(factor["alpha_annual"], float)
    assert isinstance(factor["alpha_t"], float)
    assert isinstance(factor["significant"], bool)
    assert "mkt" in factor["loadings"]
    assert "mf" not in factor["loadings"]


def test_contract_restricted_to_the_shipped_model():
    with pytest.raises(SystemExit):
        main(["--ticker", "GULFOILLUB", "--prices", str(GULFOILLUB_PRICES), "--model", "capm",
              "--contract", "/tmp/should-not-be-written.json"])
    with pytest.raises(SystemExit):
        main(["--ticker", "GULFOILLUB", "--prices", str(GULFOILLUB_PRICES), "--model", "ff3",
              "--contract", "/tmp/should-not-be-written.json"])


def test_cli_writes_contract_file(tmp_path):
    out = tmp_path / "ANYTICKER.json"
    main(["--ticker", "ANYTICKER", "--prices", str(GULFOILLUB_PRICES), "--model", "ff3wml", "--contract", str(out)])
    import json

    data = json.loads(out.read_text())
    assert set(data["factor"]) == {"alpha_annual", "alpha_t", "significant", "loadings"}


def test_cli_diagnostics_flag_runs_without_error(capsys):
    main(["--ticker", "GULFOILLUB", "--prices", str(GULFOILLUB_PRICES), "--model", "ff3wml", "--diagnostics"])
    captured = capsys.readouterr()
    assert "alpha t-stat (NW)" in captured.out
    # R11: same gap as gulfoillub.py's own CLI -- loadings_t_nw was never
    # surfaced here either.
    assert "loadings t (NW)" in captured.out


def test_cli_prints_the_sebi_disclaimer(capsys):
    # S15: same gap as gulfoillub.py's own CLI -- the disclaimer was never
    # printed, only in the static README text.
    main(["--ticker", "GULFOILLUB", "--prices", str(GULFOILLUB_PRICES), "--model", "ff3wml"])
    captured = capsys.readouterr()
    assert (
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    ) in captured.out


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        evaluate("ff5mom", prices_csv=GULFOILLUB_PRICES)


def test_cli_converts_insufficient_data_to_a_one_line_system_exit_and_writes_no_contract(tmp_path, capsys):
    # D15: a newly-listed ticker's thin price history must come back as a
    # clean one-line reason (SystemExit, no traceback), never a fabricated
    # --contract file. 8 month-end closes, all inside IIMA's covered range
    # (through 2025-12), inner-join down to 7 monthly returns -- well under
    # FF3+WML's 3x5=15-month floor.
    prices = tmp_path / "thin.csv"
    dates = [
        "2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30",
        "2025-05-31", "2025-06-30", "2025-07-31", "2025-08-31",
    ]
    rows = "\n".join(f"{d},{100 + i}" for i, d in enumerate(dates))
    prices.write_text(f"date,close\n{rows}\n")
    out = tmp_path / "should-not-be-written.json"

    with pytest.raises(SystemExit, match="too few observations"):
        main(["--ticker", "THIN", "--prices", str(prices), "--model", "ff3wml", "--contract", str(out)])

    assert capsys.readouterr().out == ""
    assert not out.exists()


def test_gulfoillub_numbers_are_identical_through_the_generic_cli(tmp_path):
    """C1/Day 22: the generalised CLI must reproduce GULFOILLUB's own pinned
    contract exactly, since STOCKSTALKER's committed ``fixtures/factor/
    GULFOILLUB.json`` was built from ``factors.gulfoillub`` and must never
    silently drift once the front door can also reach it through
    ``factors.stock``.
    """
    gulfoillub_result, _ = gulfoillub.evaluate(gulfoillub.CONTRACT_MODEL)
    stock_result, _ = evaluate(CONTRACT_MODEL, prices_csv=GULFOILLUB_PRICES)

    assert stock_result.n_obs == gulfoillub_result.n_obs
    assert stock_result.alpha_annual == pytest.approx(gulfoillub_result.alpha_annual, abs=1e-12)
    assert stock_result.alpha_t == pytest.approx(gulfoillub_result.alpha_t, abs=1e-12)
    assert stock_result.significant == gulfoillub_result.significant
    assert stock_result.loadings == pytest.approx(gulfoillub_result.loadings, abs=1e-12)

    assert to_contract(stock_result) == gulfoillub.to_contract(gulfoillub_result)

    committed = Path(__file__).resolve().parent.parent.parent / "STOCKSTALKER" / "fixtures" / "factor" / "GULFOILLUB.json"
    if committed.is_file():
        import json

        assert json.loads(committed.read_text()) == to_contract(stock_result)

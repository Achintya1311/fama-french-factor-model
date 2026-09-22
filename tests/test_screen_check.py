import json
from pathlib import Path

import pandas as pd
import pytest

from factors.screen_check import (
    DEFAULT_OHLCV_DIR,
    DEFAULT_SCREEN_PATH,
    evaluate_candidate,
    load_monthly_returns,
    load_screen,
    main,
    ohlcv_filename,
    run_screen_check,
    to_contract,
)


def test_ohlcv_filename_matches_stockstalker_convention():
    assert ohlcv_filename("RELIANCE.NS") == "RELIANCE_NS.csv"
    assert ohlcv_filename("TATACHEM.NS") == "TATACHEM_NS.csv"


def test_load_screen_returns_stockstalker_candidates_ranked():
    candidates = load_screen()
    assert [c["ticker"] for c in candidates] == ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]
    assert [c["rank"] for c in candidates] == [1, 2, 3]


def test_load_monthly_returns_matches_hand_computed_values(tmp_path):
    # Three full calendar months plus one in-progress month: the in-progress
    # one should be dropped, and the two real returns should match a plain
    # month-end-close percent change computed by hand.
    csv_path = tmp_path / "SYN_NS.csv"
    csv_path.write_text(
        "date,close,high,low,open,volume\n"
        "2024-01-05,100,100,100,100,1000\n"
        "2024-01-31,110,110,110,110,1000\n"
        "2024-02-15,90,90,90,90,1000\n"
        "2024-02-29,120,120,120,120,1000\n"
        "2024-03-10,150,150,150,150,1000\n"
    )
    returns = load_monthly_returns(csv_path)
    assert list(returns.index.astype(str)) == ["2024-02"]
    assert returns.iloc[0] == pytest.approx(120 / 110 - 1)


def test_load_monthly_returns_keeps_last_month_when_it_ends_on_month_end(tmp_path):
    csv_path = tmp_path / "SYN_NS.csv"
    csv_path.write_text(
        "date,close,high,low,open,volume\n"
        "2024-01-31,100,100,100,100,1000\n"
        "2024-02-29,110,110,110,110,1000\n"
    )
    returns = load_monthly_returns(csv_path)
    assert list(returns.index.astype(str)) == ["2024-02"]
    assert returns.iloc[0] == pytest.approx(0.10)


def test_load_monthly_returns_index_is_period_m(tmp_path):
    csv_path = tmp_path / "SYN_NS.csv"
    csv_path.write_text(
        "date,close,high,low,open,volume\n"
        "2024-01-02,100,100,100,100,1000\n"
        "2024-01-31,105,105,105,105,1000\n"
        "2024-02-29,110,110,110,110,1000\n"
    )
    returns = load_monthly_returns(csv_path)
    assert isinstance(returns.index, pd.PeriodIndex)
    assert returns.index.freqstr == "M"


def test_run_screen_check_evaluates_every_candidate_capm():
    rows = run_screen_check(DEFAULT_SCREEN_PATH, DEFAULT_OHLCV_DIR, model="capm")
    assert [r["ticker"] for r in rows] == ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]
    for row in rows:
        assert "error" not in row
        result = row["result"]
        assert result.n_obs > 0
        assert isinstance(result.significant, bool)


@pytest.mark.parametrize("model", ["capm", "ff3", "ff5mom"])
def test_run_screen_check_all_models_run_without_error(model):
    rows = run_screen_check(DEFAULT_SCREEN_PATH, DEFAULT_OHLCV_DIR, model=model)
    assert all("result" in r for r in rows)


def test_run_screen_check_records_error_for_missing_fixture(tmp_path):
    screen_path = tmp_path / "screen.json"
    screen_path.write_text(
        '{"schema": "v1", "generated": "2024-01-01", "universe": "NSE", "source": "fixtures", '
        '"candidates": [{"ticker": "NOPE.NS", "score": 0.5, "rank": 1}]}'
    )
    rows = run_screen_check(screen_path, DEFAULT_OHLCV_DIR, model="capm")
    assert len(rows) == 1
    assert "error" in rows[0]
    assert "NOPE.NS" in rows[0]["error"]


def test_screen_fixture_actually_exists():
    assert Path(DEFAULT_SCREEN_PATH).exists()
    assert Path(DEFAULT_OHLCV_DIR).is_dir()


def _reliance_result(model="ff5mom"):
    csv_path = Path(DEFAULT_OHLCV_DIR) / ohlcv_filename("RELIANCE.NS")
    returns = load_monthly_returns(csv_path)
    result, _ = evaluate_candidate(returns, model)
    return result


def test_to_contract_has_the_v0_2_shape():
    contract = to_contract(_reliance_result("ff5mom"))
    factor = contract["factor"]
    assert set(factor) == {"alpha_annual", "alpha_t", "significant", "loadings"}
    assert isinstance(factor["alpha_annual"], float)
    assert isinstance(factor["alpha_t"], float)
    assert isinstance(factor["significant"], bool)
    assert set(factor["loadings"]) == {"mkt", "smb", "hml", "rmw", "cma", "mom"}


def test_to_contract_renames_mkt_rf_to_mkt_but_keeps_other_columns():
    ff3_contract = to_contract(_reliance_result("ff3"))
    assert set(ff3_contract["factor"]["loadings"]) == {"mkt", "smb", "hml"}
    assert "mkt_rf" not in ff3_contract["factor"]["loadings"]


def test_to_contract_values_match_the_underlying_result():
    result = _reliance_result("ff5mom")
    contract = to_contract(result)["factor"]
    assert contract["alpha_annual"] == pytest.approx(result.alpha_annual)
    assert contract["alpha_t"] == pytest.approx(result.alpha_t)
    assert contract["significant"] == result.significant
    assert contract["loadings"]["mkt"] == pytest.approx(result.loadings["mkt_rf"])
    assert contract["loadings"]["smb"] == pytest.approx(result.loadings["smb"])


def test_cli_contract_writes_the_matching_candidates_contract(tmp_path, capsys):
    out_path = tmp_path / "factor.json"
    main(["--model", "ff5mom", "--contract", str(out_path), "--contract-ticker", "RELIANCE"])
    written = json.loads(out_path.read_text())
    expected = to_contract(_reliance_result("ff5mom"))
    assert written == expected
    assert "wrote factor contract for RELIANCE.NS" in capsys.readouterr().out


def test_cli_contract_matches_without_the_ns_suffix(tmp_path):
    """monte-carlo-risk-lab and reverse-dcf-engine's gates strip '.NS' too; this CLI
    does the same so a caller can pass either spelling."""
    out_path = tmp_path / "factor.json"
    main(["--model", "ff3", "--contract", str(out_path), "--contract-ticker", "tatachem"])
    written = json.loads(out_path.read_text())
    tatachem_csv = Path(DEFAULT_OHLCV_DIR) / ohlcv_filename("TATACHEM.NS")
    tatachem_result, _ = evaluate_candidate(load_monthly_returns(tatachem_csv), "ff3")
    assert written == to_contract(tatachem_result)


def test_cli_contract_requires_ticker_and_path_together():
    with pytest.raises(SystemExit):
        main(["--model", "ff5mom", "--contract", "/tmp/whatever.json"])


def test_cli_contract_rejects_capm(tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "capm", "--contract", str(tmp_path / "x.json"), "--contract-ticker", "RELIANCE"])


def test_cli_contract_rejects_a_ticker_outside_the_screen(tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "ff5mom", "--contract", str(tmp_path / "x.json"), "--contract-ticker", "NOPE"])

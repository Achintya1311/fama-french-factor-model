from pathlib import Path

import pandas as pd
import pytest

from factors.screen_check import (
    DEFAULT_OHLCV_DIR,
    DEFAULT_SCREEN_PATH,
    load_monthly_returns,
    load_screen,
    ohlcv_filename,
    run_screen_check,
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

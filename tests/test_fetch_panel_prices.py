"""Tests for scripts/fetch_panel_prices.py. All offline: no network, no keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import fetch_panel_prices  # noqa: E402

from factors.screen_check import load_monthly_returns  # noqa: E402


def _frame(dates: list[str], close: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    n = len(idx)
    return pd.DataFrame(
        {
            "Open": [close] * n,
            "High": [close + 1] * n,
            "Low": [close - 1] * n,
            "Close": [close + i * 0.1 for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


def test_write_fixture_round_trips_through_load_monthly_returns(tmp_path):
    # Real calendar month-end dates (Jan/Feb/Mar 2024), so load_monthly_returns's
    # own "drop a still-in-progress final month" guard doesn't discard the last one.
    dates = ["2024-01-31", "2024-02-29", "2024-03-31"]
    df = _frame(dates)

    path = fetch_panel_prices.write_fixture("TEST.NS", df, fixtures_dir=tmp_path)

    returns = load_monthly_returns(path)
    assert len(returns) == 2  # 3 complete month-end closes -> 2 month-over-month returns


def test_write_fixture_uses_the_same_filename_rule_as_fetch_largecaps(tmp_path):
    df = _frame(["2024-01-01", "2024-01-02"])
    path = fetch_panel_prices.write_fixture("RELIANCE.NS", df, fixtures_dir=tmp_path)
    assert path.name == "RELIANCE_NS.csv"


def test_fetch_and_write_does_not_require_a_shared_calendar(tmp_path, monkeypatch):
    # Unlike fetch_largecaps.py's intersected calendar, this panel's own
    # consumer (factors.panel) joins each ticker against IIMA independently,
    # so two tickers with non-overlapping listing histories must both still
    # get their own fixture written, not be dropped to a shared calendar.
    raw = {
        "A.NS": _frame(["2015-01-01", "2015-01-02", "2015-01-03"]),
        "B.NS": _frame(["2024-01-01", "2024-01-02"], close=50.0),
    }
    monkeypatch.setattr(fetch_panel_prices, "download_one", lambda t, start, end, max_retries=4: raw[t])

    written, skipped = fetch_panel_prices.fetch_and_write(["A.NS", "B.NS"], fixtures_dir=tmp_path)

    assert skipped == []
    assert {p.name for p in written} == {"A_NS.csv", "B_NS.csv"}
    # each fixture keeps its own full date range, not an intersection
    a_df = pd.read_csv(tmp_path / "A_NS.csv")
    assert len(a_df) == 3
    b_df = pd.read_csv(tmp_path / "B_NS.csv")
    assert len(b_df) == 2


def test_fetch_and_write_skips_a_permanently_failing_ticker_instead_of_aborting_the_whole_batch(
    tmp_path, monkeypatch
):
    raw = {
        "A.NS": _frame(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "C.NS": _frame(["2024-01-01", "2024-01-02", "2024-01-03"], close=50.0),
    }

    def fake_download_one(ticker, start, end, max_retries=4):
        if ticker == "BAD.NS":
            raise RuntimeError("BAD.NS: no usable data after 4 attempts")
        return raw[ticker]

    monkeypatch.setattr(fetch_panel_prices, "download_one", fake_download_one)

    written, skipped = fetch_panel_prices.fetch_and_write(["A.NS", "BAD.NS", "C.NS"], fixtures_dir=tmp_path)

    assert {p.name for p in written} == {"A_NS.csv", "C_NS.csv"}
    assert skipped == ["BAD.NS"]
    assert not (tmp_path / "BAD_NS.csv").exists()


def test_main_rejects_fewer_than_two_tickers(capsys):
    rc = fetch_panel_prices.main(["--tickers", "RELIANCE.NS"])
    assert rc == 1
    assert "at least 2 tickers" in capsys.readouterr().err


def test_main_defaults_to_the_fo_stock_universe_when_no_tickers_flag_given(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_panel_prices,
        "fetch_and_write",
        lambda tickers, start, end, fixtures_dir: ([], tickers),
    )
    # a bare run with no --tickers must still succeed its own argument
    # validation using the real 213-name universe, not crash on None
    rc = fetch_panel_prices.main(["--fixtures-dir", str(tmp_path)])
    assert rc == 1  # 0 written in this stub, correctly reported as a failure


def test_download_one_retries_then_raises_when_every_attempt_is_empty(monkeypatch):
    calls = []

    def always_empty(ticker, start=None, end=None, auto_adjust=True, progress=False):
        calls.append(ticker)
        return pd.DataFrame()

    monkeypatch.setattr(fetch_panel_prices.yf, "download", always_empty)
    monkeypatch.setattr(fetch_panel_prices.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="no usable data after 2 attempts"):
        fetch_panel_prices.download_one("X.NS", "2015-01-01", None, max_retries=2)

    assert len(calls) == 2

"""Tests for factors.universe (R17 panel universe)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from factors.universe import load_fo_stock_tickers


def _write_fixture(tmp_path: Path) -> Path:
    rows = [
        ["UNDERLYING", "SYMBOL", "OCT-26"],
        ["NIFTY 50", "NIFTY", "65"],
        ["NIFTY BANK", "BANKNIFTY", "30"],
        ["Derivatives on Individual Securities", "Symbol", "OCT-26"],
        ["ADANI ENERGY SOLUTION LTD", "ADANIENSOL", "675"],
        ["ADANI GREEN ENERGY LTD", "ADANIGREEN", "600"],
        ["SOME DUPLICATE ROW", "adanigreen", "600"],
        ["", "", ""],
    ]
    path = tmp_path / "fo_mktlots.csv"
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def test_drops_index_rows_keeps_only_stocks(tmp_path: Path) -> None:
    tickers = load_fo_stock_tickers(_write_fixture(tmp_path))
    assert "NIFTY.NS" not in tickers
    assert "BANKNIFTY.NS" not in tickers
    assert tickers == ["ADANIENSOL.NS", "ADANIGREEN.NS"]


def test_deduplicates_by_symbol(tmp_path: Path) -> None:
    tickers = load_fo_stock_tickers(_write_fixture(tmp_path))
    assert tickers.count("ADANIGREEN.NS") == 1


def test_missing_marker_row_raises(tmp_path: Path) -> None:
    path = tmp_path / "no_marker.csv"
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows([["UNDERLYING", "SYMBOL"], ["NIFTY 50", "NIFTY"]])
    with pytest.raises(ValueError, match="marker row"):
        load_fo_stock_tickers(path)


def test_real_committed_fixture_has_at_least_150_stocks_no_indices() -> None:
    tickers = load_fo_stock_tickers()
    assert len(tickers) >= 150
    for index_symbol in ("NIFTY.NS", "BANKNIFTY.NS", "FINNIFTY.NS", "MIDCPNIFTY.NS", "NIFTYNXT50.NS", "NIFTYFPI.NS"):
        assert index_symbol not in tickers
    assert len(tickers) == len(set(tickers))

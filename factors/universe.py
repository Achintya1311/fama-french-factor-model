"""NSE F&O stock-only universe, from STOCKSTALKER's committed lot-size snapshot (R17).

``fixtures/stockstalker/fo_mktlots.csv`` is a committed copy of
STOCKSTALKER's own ``fixtures/fo_mktlots.csv`` (NSE's F&O lot-size export) --
the same file-contract convention Day 5 already used for that repo's
screen/OHLCV fixtures, never a cross-repo import or a sibling-checkout read
at run time. The file lists 6 index underlyings (NIFTY, BANKNIFTY,
FINNIFTY, MIDCPNIFTY, NIFTYNXT50, NIFTYFPI) before a sub-header row
("Derivatives on Individual Securities,Symbol,...") that marks where the
per-stock rows start. This module returns only the stock rows -- the panel
alpha test (docs/PREREG_panel_alpha.md) is about stocks, not indices.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FO_MKTLOTS_PATH = ROOT / "fixtures" / "stockstalker" / "fo_mktlots.csv"

# The sub-header row's SYMBOL column reads "Symbol" (title case), distinct
# from the real header row's "SYMBOL" (all caps) -- this is what separates
# the index section above it from the per-stock rows below it.
STOCK_SECTION_MARKER = "Symbol"


def load_fo_stock_tickers(path: str | Path = DEFAULT_FO_MKTLOTS_PATH) -> list[str]:
    """Stock-only NSE F&O underlyings as yfinance tickers (``SYMBOL.NS``).

    Returns tickers in the file's own row order, deduplicated by symbol.
    Raises ``ValueError`` if the stock-section marker row is never found
    (the file's shape changed in a way this parser no longer understands --
    better to fail loudly than silently return an empty or index-polluted
    list).
    """
    with Path(path).open(newline="") as fh:
        rows = list(csv.reader(fh))

    start_idx: int | None = None
    for i, row in enumerate(rows):
        if len(row) >= 2 and row[1].strip() == STOCK_SECTION_MARKER:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(
            f"{path}: could not find the stock-section marker row "
            f"(column 2 == {STOCK_SECTION_MARKER!r}) -- file shape may have changed"
        )

    tickers: list[str] = []
    seen: set[str] = set()
    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        symbol = row[1].strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            tickers.append(f"{symbol}.NS")
    return tickers

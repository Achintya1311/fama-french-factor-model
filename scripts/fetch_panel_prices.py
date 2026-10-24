"""Fetch daily OHLCV fixtures for the NSE F&O stock panel (R17, PREREG_panel_alpha.md).

The panel-alpha cross-section test needs price history for the 213-name
NSE F&O stock universe (``factors.universe.load_fo_stock_tickers``), not
just GULFOILLUB's own fixture. Modeled directly on
``markowitz-portfolio-optimizer/scripts/fetch_largecaps.py`` (same free
source, ``yfinance``, ``auto_adjust=True``; same retry/backoff for a 429 or
an empty/all-NaN response) with one deliberate difference: that script
intersects every ticker onto one shared daily calendar, because
``optimizer/returns.py`` needs a single aligned returns matrix. This
panel's own consumer (``factors.panel``) resamples each ticker to monthly
returns independently and inner-joins each one against IIMA's factor
history on its own, so there is no shared-calendar requirement here, and
imposing one would needlessly drop stocks with different listing dates
down to whatever the youngest listing allows.

Meant to run from ``.github/workflows/fetch-panel-prices.yml``, triggered
by pushing this file or the workflow file, or ``workflow_dispatch`` -- not
by hand from a cloud sandbox (LESSONS.md point 4: this interactive
sandbox's own IP is not a reliable source for free-tier market data;
GitHub's hosted runners are).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "fixtures" / "panel_ohlcv"

START_DATE = "2015-01-01"

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5


def fixture_path(ticker: str, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    """Map a ticker like ``RELIANCE.NS`` to its fixture file."""
    return fixtures_dir / f"{ticker.replace('.', '_')}.csv"


def download_one(ticker: str, start: str, end: str | None, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Download one ticker's OHLCV history, retrying on a rate limit or an
    empty/all-NaN response, the same pattern fetch_largecaps.py uses."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as exc:  # yfinance raises a mix of requests/JSON errors on 429
            last_error = exc
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            df = df.dropna(subset=["Close"])
            if not df.empty:
                return df
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    detail = f" (last error: {last_error})" if last_error else ""
    raise RuntimeError(f"{ticker}: no usable data after {max_retries} attempts{detail}")


def write_fixture(ticker: str, df: pd.DataFrame, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    """Write ``date,close,high,low,open,volume`` -- the exact shape
    ``factors.screen_check.load_monthly_returns`` already reads, so the
    panel loader needs no second CSV parser."""
    path = fixture_path(ticker, fixtures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close", "high", "low", "open", "volume"])
        for ts, row in df.iterrows():
            writer.writerow(
                [
                    ts.strftime("%Y-%m-%d"),
                    round(float(row["Close"]), 4),
                    round(float(row["High"]), 4),
                    round(float(row["Low"]), 4),
                    round(float(row["Open"]), 4),
                    int(row["Volume"]),
                ]
            )
    return path


def fetch_and_write(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
) -> tuple[list[Path], list[str]]:
    """Best-effort fetch: one ticker failing after retries is skipped and
    logged, never allowed to abort the other 200+ tickers' fetch. Returns
    (written paths, skipped tickers) so the caller can report the honest
    success count rather than assuming every requested ticker landed."""
    written: list[Path] = []
    skipped: list[str] = []
    for t in tickers:
        try:
            df = download_one(t, start, end)
        except RuntimeError as exc:
            print(f"skipping {t}: {exc}", file=sys.stderr)
            skipped.append(t)
            continue
        written.append(write_fixture(t, df, fixtures_dir))
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        default=None,
        help="comma-separated NSE tickers; default: factors.universe.load_fo_stock_tickers()",
    )
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=None, help="default: today (yfinance's own default)")
    parser.add_argument("--fixtures-dir", default=str(FIXTURES_DIR))
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        from factors.universe import load_fo_stock_tickers

        tickers = load_fo_stock_tickers()

    if len(tickers) < 2:
        print("need at least 2 tickers", file=sys.stderr)
        return 1

    written, skipped = fetch_and_write(tickers, start=args.start, end=args.end, fixtures_dir=Path(args.fixtures_dir))
    for p in written:
        print(f"wrote {p}")
    print(f"fetched {len(written)}/{len(tickers)} tickers; {len(skipped)} skipped: {skipped}", file=sys.stderr)
    if len(written) < 2:
        print("fewer than 2 tickers had usable data", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

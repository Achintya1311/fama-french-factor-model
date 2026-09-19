"""Refresh fixtures/ken_french/10_Industry_Portfolios.csv (Day 2).

Companion to ``fetch_ken_french.py``: same Data Library, same zero-spend,
no-login download, same "commit the unmodified CSV" approach. This file is
the test portfolio for the CAPM/FF3/FF5 regressions - ten value- and
equal-weighted industry portfolios, a standard set of test assets, not
anything specific to Stock Stalker's NSE universe (see the README caveat on
applying US-constructed portfolios/factors to NSE names).

Not part of any loader's runtime path - `factors.industry` only reads the
committed CSV fixture, so tests and the CLI never need network access.

Usage:
    python scripts/fetch_industry_portfolios.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "ken_french" / "10_Industry_Portfolios.csv"

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "10_Industry_Portfolios_CSV.zip"
)
MEMBER = "10_Industry_Portfolios.csv"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_bytes = zf.read(MEMBER)

    OUT.write_bytes(csv_bytes)
    print(f"{OUT} refreshed ({len(csv_bytes):,} bytes) from {URL}")


if __name__ == "__main__":
    main()

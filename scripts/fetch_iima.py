"""Refresh fixtures/iima/ with IIM Ahmedabad's Indian four-factor monthly returns.

Indian counterpart to ``fetch_ken_french.py``. The Ken French factors are
USD-denominated US portfolios, which is the wrong benchmark for INR prices of
NSE stocks. IIMA's Indian Fama-French + Momentum library (Agarwalla, Jacob and
Varma) is free and needs no login. It builds SMB, HML, WML and the market
factor from CMIE Prowess data on BSE and NSE stocks, from October 1993 on.

Columns are Date (YYYY-MM), SMB, HML, WML, MF (market excess return) and RF,
all in percent per month. The first rows are NA before the market factor
starts. The file is committed unmodified.

IIMA re-releases the series with a new YYYY-MM prefix on the file name. To
update, change RELEASE to the prefix currently listed on
https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/ and rerun.

Not part of any loader's runtime path - loaders only read the committed CSV,
so tests and the CLI never need network access.

Usage:
    python scripts/fetch_iima.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = "2025-12"
NAME = f"{RELEASE}_FourFactors_and_Market_Returns_Monthly.csv"
OUT = ROOT / "fixtures" / "iima" / NAME
URL = f"https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/DATA/{NAME}"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        blob = resp.read()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(blob)
    print(f"{OUT} refreshed ({len(blob):,} bytes) from {URL}")


if __name__ == "__main__":
    main()

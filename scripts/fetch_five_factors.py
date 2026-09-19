"""Refresh fixtures/ken_french/F-F_Research_Data_5_Factors_2x3.csv (Day 3).

Companion to ``fetch_ken_french.py``: same Data Library, same zero-spend,
no-login download, same "commit the unmodified CSV" approach. Adds RMW
(profitability) and CMA (investment) to the Mkt-RF/SMB/HML/RF file already
fetched by ``fetch_ken_french.py`` - the two factors FF5 adds over FF3.

Not part of any loader's runtime path - `factors.five_factor` only reads the
committed CSV fixture, so tests and the CLI never need network access.

Usage:
    python scripts/fetch_five_factors.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "ken_french" / "F-F_Research_Data_5_Factors_2x3.csv"

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MEMBER = "F-F_Research_Data_5_Factors_2x3.csv"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_bytes = zf.read(MEMBER)

    OUT.write_bytes(csv_bytes)
    print(f"{OUT} refreshed ({len(csv_bytes):,} bytes) from {URL}")


if __name__ == "__main__":
    main()

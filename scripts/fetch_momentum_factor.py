"""Refresh fixtures/ken_french/F-F_Momentum_Factor.csv (Day 3).

Companion to ``fetch_ken_french.py``: same Data Library, same zero-spend,
no-login download, same "commit the unmodified CSV" approach. Mom is the
sixth factor FF5+momentum adds - not part of French's own FF5 file, shipped
as its own single-column download instead.

Not part of any loader's runtime path - `factors.momentum` only reads the
committed CSV fixture, so tests and the CLI never need network access.

Usage:
    python scripts/fetch_momentum_factor.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "ken_french" / "F-F_Momentum_Factor.csv"

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)
MEMBER = "F-F_Momentum_Factor.csv"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_bytes = zf.read(MEMBER)

    OUT.write_bytes(csv_bytes)
    print(f"{OUT} refreshed ({len(csv_bytes):,} bytes) from {URL}")


if __name__ == "__main__":
    main()

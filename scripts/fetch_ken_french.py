"""Refresh fixtures/ken_french/F-F_Research_Data_Factors.csv (Day 1).

Not part of the loader's runtime path - `factors.kenfrench` only reads the
committed CSV fixture, so tests and any future CLI never need network
access. This script exists so the fixture is reproducible rather than a
one-off hand-copy: re-run it to pull a fresh CSV before extending the date
range.

The Ken French Data Library is free, no login or API key needed, so this
stays inside the zero-spend rule. Stdlib only (urllib + zipfile) - nothing
new to add to requirements.txt for a script nothing at runtime imports.

Usage:
    python scripts/fetch_ken_french.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "ken_french" / "F-F_Research_Data_Factors.csv"

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
MEMBER = "F-F_Research_Data_Factors.csv"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_bytes = zf.read(MEMBER)

    OUT.write_bytes(csv_bytes)
    print(f"{OUT} refreshed ({len(csv_bytes):,} bytes) from {URL}")


if __name__ == "__main__":
    main()

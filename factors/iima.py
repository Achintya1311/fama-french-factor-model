"""Loader for IIM Ahmedabad's Indian four-factor monthly returns (Day 4).

Ken French's factors (``kenfrench.py``, ``five_factor.py``) are USD-denominated
US portfolios - the wrong benchmark for INR prices of NSE stocks. IIMA's
Indian Fama-French + Momentum library (Agarwalla, Jacob and Varma) is the
free Indian counterpart: SMB, HML, WML (momentum) and MF (market excess
return) built from CMIE Prowess data on BSE/NSE stocks, plus RF, from
1993-10 on. ``scripts/fetch_iima.py`` refreshes the committed CSV; this
module only ever reads it.

The file is a single flat table (``Date,SMB,HML,WML,MF,RF``, one header
row, no annual section like the Ken French layout) - far simpler to parse
than ``kenfrench.py``'s two-section format, but it is not NA-free the way
that assumption might suggest: the first row (1993-10) has no MF/RF because
the market factor series starts one month after SMB/HML/WML's, and one
later row (1994-11) has no SMB/HML/WML for the opposite reason (a gap in
the size/value/momentum portfolios that month, not a leading artifact).
Every row with any NA in it is dropped, not just a leading block of them.

Values in the source file are percent per month; every column returned here
is a plain decimal, matching the Ken French loaders. MF is already defined
as a market *excess* return in the source documentation (it is built the
same way Mkt-RF is, just under a different name) - it is not further
reduced by RF here, unlike a raw market return would need to be.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "iima"
DEFAULT_IIMA_PATH = FIXTURES_DIR / "2025-12_FourFactors_and_Market_Returns_Monthly.csv"

IIMA_COLUMNS = ["smb", "hml", "wml", "mf", "rf"]


def load_iima_factors(path: str | Path = DEFAULT_IIMA_PATH) -> pd.DataFrame:
    """Return SMB/HML/WML/MF/RF as decimal monthly returns.

    Indexed by a monthly :class:`pandas.PeriodIndex` named ``month``, sorted
    ascending - the same indexing convention ``kenfrench.load_monthly_factors``
    uses, so the two can be joined or swapped without touching callers.
    Rows with any ``NA`` value in the source (see module docstring) are
    dropped entirely rather than partially filled.
    """
    df = pd.read_csv(path, na_values="NA")
    df = df.dropna(subset=["SMB", "HML", "WML", "MF", "RF"])
    if df.empty:
        raise ValueError(f"no complete rows found in {path}")

    index = pd.PeriodIndex(df["Date"], freq="M", name="month")
    out = df[["SMB", "HML", "WML", "MF", "RF"]].astype(float) / 100.0
    out.columns = IIMA_COLUMNS
    out.index = index
    out = out.sort_index()

    if not out.index.is_unique:
        dupes = out.index[out.index.duplicated()].unique().tolist()
        raise ValueError(f"{path}: duplicate month(s) after parsing: {dupes}")

    return out

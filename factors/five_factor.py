"""Loader for the Ken French FF5 factor CSV (Day 3).

Same multi-section layout as ``kenfrench.py`` (header prose, a monthly
section keyed ``YYYYMM``, a blank line, an "Annual Factors" label, an annual
section keyed by year, a trailing copyright line) but with six data columns
instead of four: Mkt-RF, SMB, HML, RMW (profitability), CMA (investment),
RF. RMW and CMA are the two factors FF5 adds over FF3.

Values in the source file are percent; every column returned here is a
plain decimal, matching ``kenfrench.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ken_french"
DEFAULT_FIVE_FACTOR_PATH = FIXTURES_DIR / "F-F_Research_Data_5_Factors_2x3.csv"

FIVE_FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
_HEADER_FIELDS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def _is_header_row(fields: list[str]) -> bool:
    return fields[0] == "" and fields[1:7] == _HEADER_FIELDS


def _find_section_headers(lines: list[str]) -> list[int]:
    """Return the line indices of the two ``,Mkt-RF,...,RF`` header rows."""
    headers = [i for i, ln in enumerate(lines) if _is_header_row([f.strip() for f in ln.split(",")])]
    if len(headers) < 2:
        raise ValueError(
            f"expected 2 'Mkt-RF' header rows (monthly + annual), found {len(headers)}"
        )
    return headers


def _read_data_rows(lines: list[str], start: int, id_width: int) -> list[list[str]]:
    """Read consecutive data rows after a section header until the id field stops matching."""
    rows = []
    for ln in lines[start:]:
        fields = [f.strip() for f in ln.split(",")]
        if len(fields) != 7 or not (fields[0].isdigit() and len(fields[0]) == id_width):
            break
        rows.append(fields)
    return rows


def _to_frame(rows: list[list[str]], index) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["_id", *FIVE_FACTOR_COLUMNS])
    df = df.drop(columns="_id")
    df = df.astype(float) / 100.0
    df.index = index
    return df


def load_monthly_five_factors(path: str | Path = DEFAULT_FIVE_FACTOR_PATH) -> pd.DataFrame:
    """Return the monthly Mkt-RF/SMB/HML/RMW/CMA/RF section as decimal returns.

    Indexed by a monthly :class:`pandas.PeriodIndex` named ``month``, sorted
    ascending. Raises ``ValueError`` if the file doesn't have the two-header
    monthly/annual layout the Data Library ships.
    """
    lines = Path(path).read_text().splitlines()
    monthly_header = _find_section_headers(lines)[0]
    rows = _read_data_rows(lines, monthly_header + 1, id_width=6)
    if not rows:
        raise ValueError(f"no monthly rows found in {path}")
    index = pd.PeriodIndex([r[0] for r in rows], freq="M", name="month")
    return _to_frame(rows, index).sort_index()


def load_annual_five_factors(path: str | Path = DEFAULT_FIVE_FACTOR_PATH) -> pd.DataFrame:
    """Return the annual Mkt-RF/SMB/HML/RMW/CMA/RF section as decimal returns.

    Indexed by a yearly :class:`pandas.PeriodIndex` named ``year``, sorted
    ascending.
    """
    lines = Path(path).read_text().splitlines()
    annual_header = _find_section_headers(lines)[1]
    rows = _read_data_rows(lines, annual_header + 1, id_width=4)
    if not rows:
        raise ValueError(f"no annual rows found in {path}")
    index = pd.PeriodIndex([r[0] for r in rows], freq="Y", name="year")
    return _to_frame(rows, index).sort_index()

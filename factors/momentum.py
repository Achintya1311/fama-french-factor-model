"""Loader for the Ken French momentum factor CSV (Day 3).

Same multi-section layout as ``kenfrench.py`` and ``five_factor.py`` - a
monthly section keyed ``YYYYMM``, an annual section keyed by year, both
under a ``,Mom`` header - but French ships Mom as its own single-column
file rather than folding it into the FF5 download, so this is a separate
loader rather than an extra column on ``five_factor.py``.

Values in the source file are percent; the column returned here is a plain
decimal, matching the other loaders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ken_french"
DEFAULT_MOMENTUM_PATH = FIXTURES_DIR / "F-F_Momentum_Factor.csv"


def _is_header_row(fields: list[str]) -> bool:
    return fields == ["", "Mom"]


def _find_section_headers(lines: list[str]) -> list[int]:
    """Return the line indices of the two ``,Mom`` header rows."""
    headers = [i for i, ln in enumerate(lines) if _is_header_row([f.strip() for f in ln.split(",")])]
    if len(headers) < 2:
        raise ValueError(f"expected 2 'Mom' header rows (monthly + annual), found {len(headers)}")
    return headers


def _read_data_rows(lines: list[str], start: int, id_width: int) -> list[list[str]]:
    """Read consecutive data rows after a section header until the id field stops matching."""
    rows = []
    for ln in lines[start:]:
        fields = [f.strip() for f in ln.split(",")]
        if len(fields) != 2 or not (fields[0].isdigit() and len(fields[0]) == id_width):
            break
        rows.append(fields)
    return rows


def _to_series(rows: list[list[str]], index) -> pd.Series:
    df = pd.DataFrame(rows, columns=["_id", "mom"])
    values = df["mom"].astype(float) / 100.0
    values.index = index
    values.name = "mom"
    return values


def load_monthly_momentum(path: str | Path = DEFAULT_MOMENTUM_PATH) -> pd.Series:
    """Return the monthly Mom section as decimal returns.

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
    return _to_series(rows, index).sort_index()


def load_annual_momentum(path: str | Path = DEFAULT_MOMENTUM_PATH) -> pd.Series:
    """Return the annual Mom section as decimal returns.

    Indexed by a yearly :class:`pandas.PeriodIndex` named ``year``, sorted
    ascending.
    """
    lines = Path(path).read_text().splitlines()
    annual_header = _find_section_headers(lines)[1]
    rows = _read_data_rows(lines, annual_header + 1, id_width=4)
    if not rows:
        raise ValueError(f"no annual rows found in {path}")
    index = pd.PeriodIndex([r[0] for r in rows], freq="Y", name="year")
    return _to_series(rows, index).sort_index()

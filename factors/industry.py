"""Loader for the Ken French 10 Industry Portfolios CSV (Day 2).

Same multi-section, CRLF-terminated layout style as ``kenfrench.py`` but with
ten data columns instead of four and seven sections instead of two. This
module parses only the first section - monthly value-weighted returns - the
test portfolios the CAPM/FF3/FF5 regressions run against. The other six
sections (equal-weighted, annual, firm counts, BE/ME) are left unparsed;
nothing downstream needs them yet.

Values in the source file are percent; every column returned here is a
plain decimal, matching ``kenfrench.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ken_french"
DEFAULT_INDUSTRY_PATH = FIXTURES_DIR / "10_Industry_Portfolios.csv"

INDUSTRY_COLUMNS = [
    "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
    "Telcm", "Shops", "Hlth", "Utils", "Other",
]


def _is_header_row(fields: list[str]) -> bool:
    return fields[0] == "" and fields[1:] == INDUSTRY_COLUMNS


def _find_monthly_value_weighted_header(lines: list[str]) -> int:
    """Return the line index of the first ``,NoDur,...,Other`` header row.

    That first occurrence is always the "Average Value Weighted Returns --
    Monthly" section - the label line above it is prose, not part of the
    parse, so this only checks the header row itself.
    """
    for i, ln in enumerate(lines):
        if _is_header_row([f.strip() for f in ln.split(",")]):
            return i
    raise ValueError("no ',NoDur,Durbl,...,Other' header row found")


def _read_data_rows(lines: list[str], start: int) -> list[list[str]]:
    """Read consecutive 6-digit-YYYYMM data rows after a section header.

    A section ends at the first row that isn't a bare 6-digit id followed by
    exactly 10 fields - a blank line or the next section's label both fail
    that check.
    """
    rows = []
    for ln in lines[start:]:
        fields = [f.strip() for f in ln.split(",")]
        if len(fields) != 1 + len(INDUSTRY_COLUMNS) or not (fields[0].isdigit() and len(fields[0]) == 6):
            break
        rows.append(fields)
    return rows


def load_monthly_value_weighted(path: str | Path = DEFAULT_INDUSTRY_PATH) -> pd.DataFrame:
    """Return the monthly value-weighted industry returns as decimal returns.

    Indexed by a monthly :class:`pandas.PeriodIndex` named ``month``, sorted
    ascending. Raises ``ValueError`` if the file doesn't have the expected
    header row.
    """
    lines = Path(path).read_text().splitlines()
    header = _find_monthly_value_weighted_header(lines)
    rows = _read_data_rows(lines, header + 1)
    if not rows:
        raise ValueError(f"no monthly value-weighted rows found in {path}")
    df = pd.DataFrame(rows, columns=["_id", *INDUSTRY_COLUMNS])
    df = df.drop(columns="_id").astype(float) / 100.0
    df.index = pd.PeriodIndex([r[0] for r in rows], freq="M", name="month")
    return df.sort_index()

"""Loader for Ken French Data Library factor CSVs (Day 1).

The Data Library's CSVs are not a clean table: a few lines of header prose,
a monthly section keyed ``YYYYMM``, a blank line, an "Annual Factors" label,
an annual section keyed by 4-digit year with the *same* column header
repeated, and a trailing copyright line. This module reads that layout
directly off the committed fixture rather than a hand-cleaned copy, so
re-running ``scripts/fetch_ken_french.py`` against a fresh download never
requires touching this code.

Values in the source file are percent (e.g. ``2.89`` means 2.89%); every
column returned here is a plain decimal (``0.0289``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ken_french"
DEFAULT_FACTORS_PATH = FIXTURES_DIR / "F-F_Research_Data_Factors.csv"

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rf"]


def _is_header_row(fields: list[str]) -> bool:
    return fields[0] == "" and fields[1:5] == ["Mkt-RF", "SMB", "HML", "RF"]


def _find_section_headers(lines: list[str]) -> list[int]:
    """Return the line indices of the two ``,Mkt-RF,SMB,HML,RF`` header rows."""
    headers = [i for i, ln in enumerate(lines) if _is_header_row([f.strip() for f in ln.split(",")])]
    if len(headers) < 2:
        raise ValueError(
            f"expected 2 'Mkt-RF' header rows (monthly + annual), found {len(headers)}"
        )
    return headers


def _read_data_rows(lines: list[str], start: int, id_width: int) -> list[list[str]]:
    """Read consecutive data rows after a section header until the id field stops matching.

    A section ends at the first row whose leading field is not a bare
    ``id_width``-digit integer - a blank line, the "Annual Factors" label, or
    the trailing copyright notice all fail that check.
    """
    rows = []
    for ln in lines[start:]:
        fields = [f.strip() for f in ln.split(",")]
        if len(fields) != 5 or not (fields[0].isdigit() and len(fields[0]) == id_width):
            break
        rows.append(fields)
    return rows


def _to_frame(rows: list[list[str]], index) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["_id", *FACTOR_COLUMNS])
    df = df.drop(columns="_id")
    df = df.astype(float) / 100.0
    df.index = index
    return df


def load_monthly_factors(path: str | Path = DEFAULT_FACTORS_PATH) -> pd.DataFrame:
    """Return the monthly Mkt-RF/SMB/HML/RF section as decimal returns.

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


def load_annual_factors(path: str | Path = DEFAULT_FACTORS_PATH) -> pd.DataFrame:
    """Return the annual Mkt-RF/SMB/HML/RF section as decimal returns.

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


# RF is a real return series (cash), so compounding 12 monthly figures
# reproduces the annual one almost exactly - the tight tolerance is a genuine
# identity check. Mkt-RF/SMB/HML are zero-investment long/short portfolios
# that French reconstitutes annually each June; their *published* annual
# figure is not the compound of the 12 *published* monthly figures for that
# calendar year, so these tolerances are deliberately loose - a smoke test
# for gross parsing bugs (wrong column, forgotten /100, an off-by-one row),
# not a correctness proof. Measured empirically across all 99 complete years
# in the committed fixture: max abs gap is 2.4pp (mkt_rf), 9.2pp (smb), and
# 21.0pp (hml, in 2020 - the widest value/growth divergence in the series).
DEFAULT_COMPOUNDING_TOLERANCES = {"mkt_rf": 0.05, "smb": 0.15, "hml": 0.30, "rf": 0.001}


def annual_compounding_gaps(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    tolerances: dict[str, float] = DEFAULT_COMPOUNDING_TOLERANCES,
) -> dict[str, list[str]]:
    """Cross-check the two sections against each other, per column.

    For each column, compounds that year's 12 monthly decimal returns
    (``prod(1 + r) - 1``) and compares to the published annual figure.
    See :data:`DEFAULT_COMPOUNDING_TOLERANCES` for why RF gets a tight bound
    and the three portfolio-return columns get a loose one. Returns
    ``{column: [year, ...]}`` for any year exceeding its tolerance; an empty
    dict means nothing exceeded the (deliberately generous, for three of the
    four columns) bound - not that the sections agree closely.
    """
    monthly_by_year = monthly.groupby(monthly.index.year)
    gaps: dict[str, list[str]] = {col: [] for col in FACTOR_COLUMNS}
    for year, published in annual.iterrows():
        y = year.year
        if y not in monthly_by_year.groups:
            continue
        months = monthly_by_year.get_group(y)
        if len(months) != 12:
            continue
        compounded = (1.0 + months).prod() - 1.0
        for col in FACTOR_COLUMNS:
            if abs(compounded[col] - published[col]) > tolerances[col]:
                gaps[col].append(str(y))
    return gaps

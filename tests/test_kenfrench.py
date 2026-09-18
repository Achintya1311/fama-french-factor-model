import pandas as pd
import pytest

from factors.kenfrench import (
    FACTOR_COLUMNS,
    annual_compounding_gaps,
    load_annual_factors,
    load_monthly_factors,
)


@pytest.fixture(scope="module")
def monthly():
    return load_monthly_factors()


@pytest.fixture(scope="module")
def annual():
    return load_annual_factors()


def test_monthly_columns_and_dtype(monthly):
    assert list(monthly.columns) == FACTOR_COLUMNS
    assert all(monthly[col].dtype == float for col in FACTOR_COLUMNS)


def test_monthly_index_is_sorted_unique_periods(monthly):
    assert isinstance(monthly.index, pd.PeriodIndex)
    assert monthly.index.freqstr == "M"
    assert monthly.index.is_monotonic_increasing
    assert monthly.index.is_unique


def test_monthly_first_row_matches_known_published_value(monthly):
    # 1926-07 is the first row of the file: 2.89, -2.42, -2.75, 0.22 (percent).
    row = monthly.loc[pd.Period("1926-07", freq="M")]
    assert row["mkt_rf"] == pytest.approx(0.0289)
    assert row["smb"] == pytest.approx(-0.0242)
    assert row["hml"] == pytest.approx(-0.0275)
    assert row["rf"] == pytest.approx(0.0022)


def test_values_are_decimal_not_percent(monthly):
    # A monthly Mkt-RF swing bigger than 50% in decimal terms would mean the
    # /100 conversion was skipped somewhere.
    assert monthly["mkt_rf"].abs().max() < 0.5


def test_annual_columns_and_index(annual):
    assert list(annual.columns) == FACTOR_COLUMNS
    assert isinstance(annual.index, pd.PeriodIndex)
    # pandas has spelled the annual-December frequency "A-DEC" and "Y-DEC"
    # across versions; accept either rather than pinning to one.
    assert annual.index.freqstr in ("A-DEC", "Y-DEC")
    assert annual.index.is_monotonic_increasing


def test_annual_first_row_matches_known_published_value(annual):
    # First annual row in the file is 1927: 29.44, -3.05, -3.36, 3.12 (percent).
    row = annual.loc[pd.Period("1927", freq="Y")]
    assert row["mkt_rf"] == pytest.approx(0.2944)
    assert row["rf"] == pytest.approx(0.0312)


def test_monthly_compounds_close_to_annual(monthly, annual):
    # Cross-checks the two independently-parsed sections against each other;
    # a real parsing bug (wrong column, off-by-one row, percent left
    # unconverted) blows well past the rounding slack this tolerance covers.
    gaps = annual_compounding_gaps(monthly, annual)
    assert gaps == {col: [] for col in FACTOR_COLUMNS}


def test_missing_header_rows_raise(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("just,some,header,junk\n1,2,3,4\n")
    with pytest.raises(ValueError, match="header rows"):
        load_monthly_factors(bad)

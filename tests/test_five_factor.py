import pandas as pd
import pytest

from factors.five_factor import (
    FIVE_FACTOR_COLUMNS,
    load_annual_five_factors,
    load_monthly_five_factors,
)


@pytest.fixture(scope="module")
def monthly():
    return load_monthly_five_factors()


@pytest.fixture(scope="module")
def annual():
    return load_annual_five_factors()


def test_monthly_columns_and_dtype(monthly):
    assert list(monthly.columns) == FIVE_FACTOR_COLUMNS
    assert all(monthly[col].dtype == float for col in FIVE_FACTOR_COLUMNS)


def test_monthly_index_is_sorted_unique_periods(monthly):
    assert isinstance(monthly.index, pd.PeriodIndex)
    assert monthly.index.freqstr == "M"
    assert monthly.index.is_monotonic_increasing
    assert monthly.index.is_unique


def test_monthly_first_row_matches_known_published_value(monthly):
    # 1963-07 is the first row of the file: -0.39, -0.48, -0.84, 0.64, -1.15, 0.27 (percent).
    row = monthly.loc[pd.Period("1963-07", freq="M")]
    assert row["mkt_rf"] == pytest.approx(-0.0039)
    assert row["smb"] == pytest.approx(-0.0048)
    assert row["hml"] == pytest.approx(-0.0084)
    assert row["rmw"] == pytest.approx(0.0064)
    assert row["cma"] == pytest.approx(-0.0115)
    assert row["rf"] == pytest.approx(0.0027)


def test_values_are_decimal_not_percent(monthly):
    assert monthly["mkt_rf"].abs().max() < 0.5


def test_annual_columns_and_index(annual):
    assert list(annual.columns) == FIVE_FACTOR_COLUMNS
    assert isinstance(annual.index, pd.PeriodIndex)
    assert annual.index.freqstr in ("A-DEC", "Y-DEC")
    assert annual.index.is_monotonic_increasing


def test_annual_first_row_matches_known_published_value(annual):
    # First annual row in the file is 1964: 12.59, 0.52, 9.72, -2.76, 6.49, 3.54 (percent).
    row = annual.loc[pd.Period("1964", freq="Y")]
    assert row["mkt_rf"] == pytest.approx(0.1259)
    assert row["rmw"] == pytest.approx(-0.0276)
    assert row["cma"] == pytest.approx(0.0649)


def test_missing_header_rows_raise(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("just,some,header,junk\n1,2,3,4\n")
    with pytest.raises(ValueError, match="header rows"):
        load_monthly_five_factors(bad)

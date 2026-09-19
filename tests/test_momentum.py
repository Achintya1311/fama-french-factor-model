import pandas as pd
import pytest

from factors.momentum import load_annual_momentum, load_monthly_momentum


@pytest.fixture(scope="module")
def monthly():
    return load_monthly_momentum()


@pytest.fixture(scope="module")
def annual():
    return load_annual_momentum()


def test_monthly_dtype(monthly):
    assert monthly.dtype == float
    assert monthly.name == "mom"


def test_monthly_index_is_sorted_unique_periods(monthly):
    assert isinstance(monthly.index, pd.PeriodIndex)
    assert monthly.index.freqstr == "M"
    assert monthly.index.is_monotonic_increasing
    assert monthly.index.is_unique


def test_monthly_first_row_matches_known_published_value(monthly):
    # 1927-01 is the first row of the file: 0.57 (percent).
    assert monthly.loc[pd.Period("1927-01", freq="M")] == pytest.approx(0.0057)


def test_values_are_decimal_not_percent(monthly):
    # A monthly swing bigger than 100% in decimal terms would mean the /100
    # conversion was skipped somewhere; momentum crashes (its worst months
    # are momentum-crash events) are noisier than mkt_rf, so this bound is
    # looser than kenfrench's, same reasoning as industry.py's test.
    assert monthly.abs().max() < 1.0


def test_annual_index(annual):
    assert isinstance(annual.index, pd.PeriodIndex)
    assert annual.index.freqstr in ("A-DEC", "Y-DEC")
    assert annual.index.is_monotonic_increasing


def test_annual_first_row_matches_known_published_value(annual):
    # First annual row in the file is 1927: 24.52 (percent).
    assert annual.loc[pd.Period("1927", freq="Y")] == pytest.approx(0.2452)


def test_missing_header_rows_raise(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("just,some,header,junk\n1,2,3,4\n")
    with pytest.raises(ValueError, match="header rows"):
        load_monthly_momentum(bad)

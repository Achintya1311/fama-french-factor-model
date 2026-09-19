import pandas as pd
import pytest

from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted


@pytest.fixture(scope="module")
def monthly():
    return load_monthly_value_weighted()


def test_columns_and_dtype(monthly):
    assert list(monthly.columns) == INDUSTRY_COLUMNS
    assert all(monthly[col].dtype == float for col in INDUSTRY_COLUMNS)


def test_index_is_sorted_unique_periods(monthly):
    assert isinstance(monthly.index, pd.PeriodIndex)
    assert monthly.index.freqstr == "M"
    assert monthly.index.is_monotonic_increasing
    assert monthly.index.is_unique


def test_first_row_matches_known_published_value(monthly):
    # 1926-07 is the first row of the value-weighted monthly section:
    # 1.44, 15.58, 4.70, -1.14, 2.90, 0.83, 0.12, 1.85, 7.04, 2.14 (percent).
    row = monthly.loc[pd.Period("1926-07", freq="M")]
    assert row["NoDur"] == pytest.approx(0.0144)
    assert row["Durbl"] == pytest.approx(0.1558)
    assert row["Other"] == pytest.approx(0.0214)


def test_values_are_decimal_not_percent(monthly):
    # A monthly swing bigger than 200% in decimal terms would mean the /100
    # conversion was skipped somewhere (industry portfolios are noisier than
    # the market factor, so this bound is looser than kenfrench's).
    assert monthly.abs().max().max() < 2.0


def test_stops_at_next_section_not_past_it(monthly):
    # The value-weighted monthly section is immediately followed by the
    # equal-weighted monthly section, which repeats the same header and the
    # same first id (192607). If the reader didn't stop at the blank line
    # separating sections, this index would have duplicate periods.
    assert monthly.index.is_unique


def test_missing_header_row_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("just,some,header,junk\n1,2,3,4\n")
    with pytest.raises(ValueError, match="header row"):
        load_monthly_value_weighted(bad)

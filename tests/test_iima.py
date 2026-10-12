import pandas as pd
import pytest

from factors.iima import IIMA_COLUMNS, load_iima_factors


@pytest.fixture(scope="module")
def iima():
    return load_iima_factors()


def test_columns_and_dtype(iima):
    assert list(iima.columns) == IIMA_COLUMNS
    assert all(iima[col].dtype == float for col in IIMA_COLUMNS)


def test_index_is_sorted_unique_monthly_periods(iima):
    assert isinstance(iima.index, pd.PeriodIndex)
    assert iima.index.freqstr == "M"
    assert iima.index.is_monotonic_increasing
    assert iima.index.is_unique


def test_leading_and_mid_series_na_rows_are_dropped(iima):
    # 1993-10 (the file's first row) has no MF/RF; 1994-11 has no SMB/HML/WML.
    # Both are dropped entirely, not partially filled with 0 or forward-filled.
    assert pd.Period("1993-10", freq="M") not in iima.index
    assert pd.Period("1994-11", freq="M") not in iima.index
    # The row right after the leading gap is the first row this loader keeps.
    assert iima.index.min() == pd.Period("1993-11", freq="M")


def test_first_kept_row_matches_known_published_value():
    # 1993-11 in the committed fixture: SMB=-9.41834388338565, HML=-0.565741069716541,
    # WML=12.6023590875531, MF=17.6218944309642, RF=0.611432097208131 (percent).
    iima = load_iima_factors()
    row = iima.loc[pd.Period("1993-11", freq="M")]
    assert row["smb"] == pytest.approx(-0.0941834388338565)
    assert row["hml"] == pytest.approx(-0.00565741069716541)
    assert row["wml"] == pytest.approx(0.126023590875531)
    assert row["mf"] == pytest.approx(0.176218944309642)
    assert row["rf"] == pytest.approx(0.00611432097208131)


def test_last_row_matches_known_published_value(iima):
    # 2025-12, the last row of the committed fixture.
    row = iima.loc[pd.Period("2025-12", freq="M")]
    assert row["smb"] == pytest.approx(-0.0362188737451088)
    assert row["hml"] == pytest.approx(0.0185294286104356)
    assert row["wml"] == pytest.approx(0.0139859838899936)
    assert row["mf"] == pytest.approx(-0.00720720181562362)
    assert row["rf"] == pytest.approx(0.0046772133559092)


def test_values_are_decimal_not_percent(iima):
    # A monthly move bigger than 50% in decimal terms would mean the /100
    # conversion was skipped somewhere.
    assert iima[IIMA_COLUMNS].abs().max().max() < 0.5


def test_row_count_drops_exactly_the_two_na_rows():
    # 387 data rows in the committed fixture, 2 with an NA in any column.
    iima = load_iima_factors()
    assert len(iima) == 385


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_iima_factors(tmp_path / "nope.csv")

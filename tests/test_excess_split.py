"""P140: the mean excess return split into alpha and factor exposure.

The site's factor card says GULFOILLUB's 1.30 pp mean monthly excess return
splits into 0.85 alpha and 0.45 exposure; no command printed that. With an
intercept, OLS makes the mean residual zero, so mean(excess) = alpha +
sum(loading x mean factor) exactly.
"""

import numpy as np
import pandas as pd
import pytest

from factors import gulfoillub, stock
from factors.multifactor import excess_return_split, run_factor_model

COLS = ("mf", "smb")


def _synthetic():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2010-01", periods=120, freq="M").astype(str)
    f = pd.DataFrame({"mf": rng.normal(0.01, 0.05, 120), "smb": rng.normal(0.002, 0.03, 120), "rf": 0.005}, index=idx)
    port = 0.004 + 0.9 * f["mf"] + 0.5 * f["smb"] + f["rf"] + rng.normal(0, 0.02, 120)
    return port, f


def test_parts_add_to_the_mean_exactly():
    port, f = _synthetic()
    res = run_factor_model(port, f, COLS)
    split = excess_return_split(port, f, res)
    assert split["mean_excess"] == pytest.approx(float((port - f["rf"]).mean()), abs=1e-15)
    assert split["alpha"] == pytest.approx(res.alpha_monthly, abs=1e-15)
    assert split["alpha"] + split["exposure"] == pytest.approx(split["mean_excess"], abs=1e-12)
    assert sum(split["by_factor"].values()) == pytest.approx(split["exposure"], abs=1e-15)
    assert split["by_factor"]["mf"] == pytest.approx(res.loadings["mf"] * f["mf"].mean(), abs=1e-15)


def test_no_noise_known_answer():
    _, f = _synthetic()
    port = 0.003 + 1.2 * f["mf"] - 0.4 * f["smb"] + f["rf"]
    res = run_factor_model(port, f, COLS)
    split = excess_return_split(port, f, res)
    assert split["alpha"] == pytest.approx(0.003, abs=1e-12)
    assert split["by_factor"]["mf"] == pytest.approx(1.2 * f["mf"].mean(), abs=1e-12)
    assert split["by_factor"]["smb"] == pytest.approx(-0.4 * f["smb"].mean(), abs=1e-12)


def test_gulfoillub_fixture_split_matches_the_site_card(capsys):
    gulfoillub.main(["--model", "ff3wml", "--split"])
    out = capsys.readouterr().out
    assert "mean excess return (monthly): +1.30%" in out
    assert "alpha                       : +0.85%" in out
    assert "factor exposure             : +0.45%" in out


def test_stock_cli_prints_the_split(capsys):
    stock.main(["--ticker", "GULFOILLUB", "--prices", str(gulfoillub.DEFAULT_OHLCV_PATH), "--split"])
    assert "mean excess return (monthly)" in capsys.readouterr().out

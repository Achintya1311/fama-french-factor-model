"""P128 (CRITIQUES.md): measure which loadings move alpha between two models."""

import numpy as np
import pandas as pd
import pytest

from factors.decay_report import _restrict, main
from factors.industry import load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import FF3_COLUMNS, FF5_MOM_COLUMNS, alpha_change_attribution, load_ff5_mom_factors


def test_attribution_is_exact_on_one_factor_file():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2000-01", periods=240, freq="M")
    f = pd.DataFrame(rng.normal(0.005, 0.03, (240, 3)), index=idx, columns=["mkt_rf", "smb", "hml"])
    f["rf"] = 0.001
    port = 0.002 + 1.1 * f["mkt_rf"] - 0.4 * f["hml"] + 0.2 * f["smb"] + f["rf"] + rng.normal(0, 0.01, 240)
    r = alpha_change_attribution(port, f, ("mkt_rf",), f, ("mkt_rf", "smb", "hml"))
    assert sum(r["contributions"].values()) + r["residual"] == pytest.approx(r["alpha_change"], abs=1e-12)
    assert r["residual"] == pytest.approx(0.0, abs=1e-12)
    assert set(r["contributions"]) == {"mkt_rf", "smb", "hml"}


def test_hitec_ff3_to_ff5mom_split_on_the_matched_window():
    vw = _restrict(load_monthly_value_weighted(), "1963-07")["HiTec"]
    r = alpha_change_attribution(
        vw, _restrict(load_monthly_factors(), "1963-07"), FF3_COLUMNS,
        _restrict(load_ff5_mom_factors(), "1963-07"), FF5_MOM_COLUMNS,
    )
    c = {k: v * 12 for k, v in r["contributions"].items()}
    assert c["cma"] == pytest.approx(0.0120, abs=5e-4)
    assert c["rmw"] == pytest.approx(0.0096, abs=5e-4)
    assert c["hml"] < 0
    assert r["alpha_change"] * 12 == pytest.approx(0.0265, abs=5e-4)
    assert sum(r["contributions"].values()) + r["residual"] == pytest.approx(r["alpha_change"], abs=1e-12)


def test_cli_prints_the_attribution(capsys):
    main(["--since", "1963-07", "--attribution", "HiTec"])
    out = capsys.readouterr().out.split("alpha-change attribution", 1)[1]
    assert "cma" in out and "residual" in out

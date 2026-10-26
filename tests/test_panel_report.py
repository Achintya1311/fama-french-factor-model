"""D37: the committed panel report must match what factors.panel computes on
the committed fixtures. The weekly data refresh regenerates both together, so a
refresh that moves the panel result shows up in the report, never silently."""

import json
from pathlib import Path

import pytest

from factors import panel

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "panel_alpha_latest.json"


def test_committed_report_matches_the_committed_fixtures(tmp_path):
    out = tmp_path / "panel.json"
    assert panel.main(["--json", str(out)]) == 0
    fresh = json.loads(out.read_text())
    committed = json.loads(REPORT.read_text())
    assert set(fresh) == set(committed)
    for key, value in committed.items():
        if isinstance(value, float):
            assert fresh[key] == pytest.approx(value, abs=1e-6), key
        else:
            assert fresh[key] == value, key


def test_report_carries_the_headline_fields():
    committed = json.loads(REPORT.read_text())
    for key in ("n_tickers", "n_panel", "n_skipped", "fdr", "bh_rejected", "raw_significant",
                "r2_mean", "r2_median", "mde_median", "mde_mean", "grs_status", "fixture_last_date"):
        assert key in committed
    assert committed["n_panel"] + committed["n_skipped"] == committed["n_tickers"]

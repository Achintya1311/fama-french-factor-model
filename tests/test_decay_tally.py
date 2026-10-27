"""P127 (CRITIQUES.md): count, across all ten industries, whether |alpha|
shrinks or grows from CAPM to FF5+Mom on the matched window."""

from pathlib import Path

from factors.decay_report import _rows, decay_tally, main

ROOT = Path(__file__).resolve().parents[1]


def _row(industry, model, alpha, n=757):
    return {"industry": industry, "model": model, "alpha_annual": alpha, "n_obs": n}


def test_tally_on_synthetic_rows():
    rows = [
        _row("A", "CAPM", 0.02), _row("A", "FF5+Mom", -0.01),
        _row("B", "CAPM", -0.01), _row("B", "FF5+Mom", 0.03),
        _row("C", "CAPM", 0.01), _row("C", "FF5+Mom", 0.01),
    ]
    t = decay_tally(rows)
    assert t["shrinks"] == ["A"]
    assert t["grows"] == ["B"]
    assert t["unchanged"] == ["C"]
    assert t["matched_window"] is True


def test_tally_flags_a_mismatched_window():
    rows = [_row("A", "CAPM", 0.02, n=1201), _row("A", "FF5+Mom", 0.01)]
    assert decay_tally(rows)["matched_window"] is False


def test_fixture_tally_on_the_matched_window():
    t = decay_tally(_rows("1963-07"))
    assert t["matched_window"] is True
    assert sorted(t["shrinks"]) == sorted(["NoDur", "Durbl", "Telcm", "Shops", "Hlth", "Utils"])
    assert sorted(t["grows"]) == sorted(["Manuf", "Enrgy", "HiTec", "Other"])


def test_cli_prints_the_tally(capsys):
    main(["--since", "1963-07"])
    assert "|alpha| shrinks CAPM -> FF5+Mom in 6 of 10 industries" in capsys.readouterr().out


def test_lead_finding_carries_the_tally():
    lead = next(l for l in (ROOT / "README.md").read_text().splitlines() if l.startswith("**Finding"))
    assert "6 of 10" in lead

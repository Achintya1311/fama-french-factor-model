"""R23: the README's lead Finding matches decay_report on both windows."""

from pathlib import Path

from factors.decay_report import _rows

ROOT = Path(__file__).resolve().parents[1]


def _lead() -> str:
    text = (ROOT / "README.md").read_text()
    return next(line for line in text.splitlines() if line.startswith("**Finding"))


def _hitec(since: str | None) -> dict[str, float]:
    return {r["model"]: r["alpha_annual"] for r in _rows(since) if r["industry"] == "HiTec"}


def test_lead_quotes_hitec_on_the_full_and_the_matched_window():
    lead = _lead()
    for since in (None, "1963-07"):
        for model, alpha in _hitec(since).items():
            assert f"{alpha:+.2%}" in lead, (since, model)


def test_lead_no_longer_says_the_growth_is_only_the_window():
    assert "not more factors" not in _lead()

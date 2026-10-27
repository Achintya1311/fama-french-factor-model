"""P105, P106: every HiTec alpha t-stat the README quotes says which standard error it is.

The labelled figures are recomputed from ``factors.decay_report``'s own rows,
so a README number and its label cannot drift from the command that made it.
"""

from pathlib import Path

import pytest

from factors.decay_report import _rows

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _t(since, model):
    row = next(r for r in _rows(since) if r["industry"] == "HiTec" and r["model"] == model)
    return f"{row['alpha_t']:.2f}", f"{row['alpha_t_nw']:.2f}"


@pytest.mark.parametrize(
    "since,model,expect",
    [
        (None, "FF3", ("1.55", "1.43")),
        ("1963-07", "CAPM", ("0.19", "0.17")),
        ("1963-07", "FF3", ("2.21", "2.17")),
        ("1963-07", "FF5+Mom", ("4.47", "4.32")),
    ],
)
def test_quoted_hitec_tstats_match_decay_report(since, model, expect):
    assert _t(since, model) == expect


def test_window_isolation_paragraph_labels_both_bases():
    for text in (
        "+0.26%, t=0.19 OLS / 0.17 NW",
        "+1.51%, t=1.55 OLS / 1.43 NW",
        "+2.69%, t=2.21 OLS / 2.17 NW",
        "+5.44%, t=4.47 OLS / 4.32 NW",
    ):
        assert text in README


def test_no_unlabelled_mixed_comparison_remains():
    # The checkpoint log is history and keeps its original wording.
    body = README.split("<!-- CHECKPOINTS:START -->", 1)[0]
    assert "(t=1.55 -> t=2.17)" not in body
    assert "OLS 1.55 -> 2.21, Newey-West 1.43 -> 2.17" in README
    assert "t = 4.47 OLS" in README


def test_checkpoint_rows_carry_their_commits():
    log = README.split("<!-- CHECKPOINTS:START -->", 1)[1].split("<!-- CHECKPOINTS:END -->", 1)[0]
    assert "(pending)" not in log
    assert "| 2026-10-18 | `a4ca749` |" in log
    assert "| 2026-10-12 | `cf385b3` |" in log

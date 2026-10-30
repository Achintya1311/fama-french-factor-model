"""P153/P155: the README's gate and hand-off say what the code does."""
from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()


def _section(title: str) -> str:
    start = README.index(f"## {title}\n")
    end = README.index("\n## ", start + 1)
    return README[start:end]


def test_correctness_gate_does_not_claim_a_published_replication():
    gate = _section("Correctness gate")
    assert "Replicates a published FF3 result" not in gate
    assert "Exact-identity" in gate
    assert "tests/test_multifactor.py" in gate
    assert "no published FF3 figure is replicated" in gate


def test_where_this_sits_names_the_live_handoff():
    sits = _section("Where this sits")
    first = sits.split("```json")[0]
    assert "factors.stock" in first and "--model ff3wml --contract" in first
    assert "IIMA" in first
    assert "stockstalker/live_contracts.py" in first
    assert "older" in first and "factors.screen_check --contract" in first

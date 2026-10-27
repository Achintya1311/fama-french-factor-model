"""D55: the files copied from STOCKSTALKER are pinned by hash.

The OHLCV copies are also compared with the sibling checkout when one
exists. The lot list is deliberately the 2026-10-17 snapshot the panel
was pre-registered on, so it is pinned but not compared: STOCKSTALKER
has refreshed its own copy since.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COPIES = ROOT / "fixtures" / "stockstalker"
SIBLING = ROOT.parent / "STOCKSTALKER" / "fixtures"
OHLCV = {
    "RELIANCE_NS.csv": ("81d022cb3dff50dc316d94dc1a58b64947a991850c6595480165a2723b623369", "d9fe2d8"),
    "TATACHEM_NS.csv": ("986700d1bd15887486c77d240cbcd7f668664efc9d4a207b47de898e47a585ed", "d9fe2d8"),
    "CROMPTON_NS.csv": ("db66db9aba5594544631f8887e7540101257e953cdec5f65ef76318b5a54327a", "d9fe2d8"),
    "GULFOILLUB_NS.csv": ("91f0f915352e5f06bd118484fb4cec7cd8dce964d72d5c4ffc14d782d1f8d7b3", "2065467"),
}
OTHER = {
    "fo_mktlots.csv": "a94b41b6e42a2a7ce898df7c50b08c4843ebe24664145826a9b0652733d17452",
    "fo_mktlots_meta.json": "aeb5c4ed987983b544c1f9edf54e806349fd5bad891ec6824dccf3679ce164ed",
    "screen_2026-09-20.json": "c1c30b0ab2887170810df496f23f28298005e7f6afad112c81ac5d56e9a8c036",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_sources() -> str:
    text = (ROOT / "README.md").read_text()
    return text.split("## Data sources", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize("name", sorted(OHLCV))
def test_ohlcv_copy_matches_the_pinned_hash(name):
    assert _sha256(COPIES / "ohlcv" / name) == OHLCV[name][0]


@pytest.mark.parametrize("name", sorted(OTHER))
def test_other_copy_matches_the_pinned_hash(name):
    assert _sha256(COPIES / name) == OTHER[name]


@pytest.mark.parametrize("name", sorted(OHLCV))
def test_ohlcv_copy_matches_stockstalker_when_checked_out(name):
    sibling = SIBLING / "ohlcv" / name
    if not sibling.is_file():
        pytest.skip("STOCKSTALKER is not checked out next to this repo")
    assert (COPIES / "ohlcv" / name).read_bytes() == sibling.read_bytes()


def test_data_sources_name_every_copy_its_hash_and_source():
    sources = _data_sources()
    for sha, commit in OHLCV.values():
        assert sha in sources and commit in sources
    for sha in OTHER.values():
        assert sha in sources
    for token in ("69a8de7", "43e8cdd", "ac11565", "auto_adjust=True", "back-adjusted",
                  "fixtures/panel_ohlcv/", "fetch-panel-prices.yml"):
        assert token in sources

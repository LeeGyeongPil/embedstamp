import pytest

from embedstamp import FakeEmbedder, LocalIndex, stamp
from embedstamp.demo import build_index


@pytest.fixture(autouse=True)
def _clean_language_env(monkeypatch):
    for var in ("EMBEDSTAMP_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def stored_a():
    """Fingerprint of the fake:a model."""
    return stamp(FakeEmbedder("fake:a"), model="fake:a")


@pytest.fixture
def demo_index(tmp_path) -> LocalIndex:
    """The 160-document demo corpus indexed and stamped with fake:a."""
    return build_index(tmp_path / "demo", FakeEmbedder("fake:a"), "fake:a")

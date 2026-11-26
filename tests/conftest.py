import importlib
import os

import pytest

from embedstamp import FakeEmbedder, LocalIndex, stamp
from embedstamp.demo import build_index


def pytest_collection_modifyitems(config, items):
    # chromadb 픽스처를 쓰는 테스트에 chroma 마커가 빠지면 CI extras 잡(-m chroma)이 그 테스트를 놓친다 — 수집 단계에서 막는다
    for item in items:
        if "chromadb" in getattr(item, "fixturenames", ()) and item.get_closest_marker("chroma") is None:
            raise pytest.UsageError(f"{item.nodeid} uses the chromadb fixture but is not marked chroma")


@pytest.fixture
def chromadb():
    """The ``chromadb`` module, for tests marked ``chroma``.

    Without the ``chroma`` extra (the default ``uv sync``) these tests are skipped. In the CI
    extras job ``EMBEDSTAMP_REQUIRE_CHROMA=1`` turns a missing or broken chromadb into a failure,
    so a skip can never pass for a green Chroma check.
    """
    if os.environ.get("EMBEDSTAMP_REQUIRE_CHROMA") == "1":
        # 건너뛰지 않는다 — import 오류가 그대로 테스트 실패가 된다
        return importlib.import_module("chromadb")
    return pytest.importorskip("chromadb", reason="needs the chroma extra: uv run --extra chroma pytest")


@pytest.fixture
def chroma_settings(chromadb):
    """Chroma client settings with telemetry off (tests never send usage data)."""
    return importlib.import_module("chromadb.config").Settings(anonymized_telemetry=False)


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

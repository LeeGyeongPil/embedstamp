"""Regression for F-1: names that need an extra are importable from the top-level package."""

import subprocess
import sys

import pytest

import embedstamp


@pytest.fixture
def fresh_lazy_names(monkeypatch):
    # __getattr__ 가 한 번 불러온 이름은 모듈 전역에 캐시되므로, 테스트마다 지워서 다시 거치게 한다
    for name in ("ChromaStore", "FastEmbedEmbedder"):
        monkeypatch.delitem(embedstamp.__dict__, name, raising=False)


def test_chroma_store_is_importable_from_the_package(fresh_lazy_names):
    pytest.importorskip("chromadb")
    from embedstamp import ChromaStore
    from embedstamp.stores.chroma import ChromaStore as Direct

    assert ChromaStore is Direct
    assert "ChromaStore" in dir(embedstamp)


def test_chroma_store_without_extra_says_how_to_install(fresh_lazy_names, monkeypatch):
    # sys.modules 에 None 을 넣으면 `import chromadb` 가 ImportError 를 낸다(extra 가 없는 환경 흉내)
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(ImportError, match=r"embedstamp\[chroma\]"):
        from embedstamp import ChromaStore  # noqa: F401


def test_fastembed_embedder_without_extra_says_how_to_install(fresh_lazy_names, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError, match=r"embedstamp\[fastembed\]"):
        from embedstamp import FastEmbedEmbedder  # noqa: F401


def test_unknown_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError):
        embedstamp.DoesNotExist  # noqa: B018
    with pytest.raises(ImportError, match="cannot import name"):
        from embedstamp import DoesNotExist  # noqa: F401


def test_plain_import_does_not_load_optional_dependencies():
    # import embedstamp 만으로 무거운 chromadb·fastembed 를 불러오지 않는다(별도 프로세스에서 확인)
    code = (
        "import sys, embedstamp; "
        "print('chromadb' in sys.modules, 'fastembed' in sys.modules); "
        "from embedstamp import *"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]

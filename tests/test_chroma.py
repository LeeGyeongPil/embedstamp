"""Chroma integration (extra ``chroma``). Skipped without chromadb; see the ``chromadb`` fixture."""

import warnings

import pytest

from embedstamp import EmbeddingMismatchError, FakeEmbedder, Fingerprint, Guard, stamp
from embedstamp.stores.chroma import ChromaStore

# 이 파일의 모든 테스트는 chromadb 가 필요하다 — 코어 환경(uv sync)에서는 건너뛰고, CI 필수 잡에서 빠진다
pytestmark = pytest.mark.chroma

DOCS = ["Reset your password in settings.", "Invoices are due in thirty days.", "Plants make sugar."]


@pytest.fixture
def open_client(chromadb, chroma_settings):
    """Open (or re-open) a persistent Chroma database in a directory."""
    return lambda path: chromadb.PersistentClient(path=str(path), settings=chroma_settings)


@pytest.fixture
def collection(tmp_path, open_client):
    client = open_client(tmp_path)
    col = client.get_or_create_collection(
        "docs", metadata={"hnsw:space": "cosine", "owner": "team-x"}, embedding_function=None
    )
    col.add(ids=["1", "2", "3"], documents=DOCS, embeddings=FakeEmbedder("fake:a")(DOCS).tolist())
    return col


def test_fingerprint_is_stored_in_metadata_and_keeps_existing_keys(collection, tmp_path, open_client):
    store = ChromaStore(collection)
    assert store.load() is None
    fp = stamp(FakeEmbedder("fake:a"), model="fake:a")
    store.save(fp)

    assert "embedstamp" in collection.metadata
    assert collection.metadata["owner"] == "team-x"
    # the distance function lives on in the collection configuration
    assert collection.configuration["hnsw"]["space"] == "cosine"

    reopened = open_client(tmp_path).get_collection("docs", embedding_function=None)
    loaded = ChromaStore(reopened).load()
    assert isinstance(loaded, Fingerprint)
    assert loaded == fp


def test_saving_twice_replaces_only_the_fingerprint(collection):
    store = ChromaStore(collection)
    store.save(stamp(FakeEmbedder("fake:a"), model="fake:a"))
    store.save(stamp(FakeEmbedder("fake:b"), model="fake:b"))
    assert store.load().model == "fake:b"
    assert collection.metadata["owner"] == "team-x"


def test_guard_blocks_other_model_and_passes_same_model(collection):
    store = ChromaStore(collection)
    store.save(stamp(FakeEmbedder("fake:a"), model="fake:a"))

    # without a guard, Chroma answers a same-dimension query from the wrong model silently
    wrong = FakeEmbedder("fake:b")(["reset password"])[0].tolist()
    assert collection.query(query_embeddings=[wrong], n_results=2)["ids"][0]

    with pytest.raises(EmbeddingMismatchError):
        Guard(store, FakeEmbedder("fake:b"), model="fake:b").embed_query("reset password")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vec = Guard(store, FakeEmbedder("fake:a"), model="fake:a").embed_query("reset password")
    assert collection.query(query_embeddings=[vec.tolist()], n_results=1)["ids"][0] == ["1"]


def test_compact_fingerprint_is_small_enough_for_metadata(collection):
    store = ChromaStore(collection)
    store.save(stamp(FakeEmbedder("fake:a"), model="fake:a", compact=True))
    assert len(collection.metadata["embedstamp"]) <= 2500

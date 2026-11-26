import warnings

import pytest

from embedstamp import (
    EmbeddingMismatchError,
    FakeEmbedder,
    FingerprintNotFoundError,
    Guard,
    SidecarStore,
)


@pytest.fixture
def store(tmp_path, stored_a):
    s = SidecarStore(tmp_path / "fp.json")
    s.save(stored_a)
    return s


def test_raise_happens_before_the_query_is_embedded(store):
    embed = FakeEmbedder("fake:b")
    guard = Guard(store, embed, model="fake:b")
    with pytest.raises(EmbeddingMismatchError) as info:
        guard.embed_query("hello")
    assert info.value.verdict.status == "different"
    assert embed.texts_embedded == 8  # only the probe sentences, never the query


def test_warn_mode_only_warns(store):
    guard = Guard(store, FakeEmbedder("fake:b"), model="fake:b", on_mismatch="warn")
    with pytest.warns(UserWarning, match="different"):
        vec = guard.embed_query("hello")
    assert vec.shape == (384,)


def test_same_model_passes_silently(store):
    guard = Guard(store, FakeEmbedder("fake:a"), model="fake:a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert guard.embed_documents(["a", "b"]).shape == (2, 384)


def test_check_is_cached(store):
    embed = FakeEmbedder("fake:a")
    guard = Guard(store, embed, model="fake:a")
    first = guard.check()
    calls = embed.calls
    assert guard.check() is first
    assert embed.calls == calls


def test_drift_warns_or_raises_by_setting(store):
    with pytest.warns(UserWarning, match="drift"):
        Guard(store, FakeEmbedder("fake:a", noise=0.05), model="fake:a").check()
    strict = Guard(store, FakeEmbedder("fake:a", noise=0.05), model="fake:a", allow_drift=False)
    with pytest.raises(EmbeddingMismatchError):
        strict.check()


def test_prefixes_are_applied(store):
    embed = FakeEmbedder("fake:a")
    guard = Guard(store, embed, model="fake:a", query_prefix="query: ", on_mismatch="warn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = guard.embed_query("hello")
    assert (got == embed(["query: hello"])[0]).all()


def test_missing_fingerprint_explains_the_fix(tmp_path):
    guard = Guard(SidecarStore(tmp_path / "none.json"), FakeEmbedder("fake:a"), model="fake:a")
    with pytest.raises(FingerprintNotFoundError, match="stamp"):
        guard.check()


def test_invalid_on_mismatch_value(store):
    with pytest.raises(ValueError, match="on_mismatch"):
        Guard(store, FakeEmbedder("fake:a"), model="fake:a", on_mismatch="ignore")  # type: ignore[arg-type]

import pytest

from embedstamp import FakeEmbedder, Thresholds, resolve_embedder, stamp, verify


def run(stored, spec, **kw):
    embed, model = resolve_embedder(spec)
    return verify(stored, embed, model=model, **kw)


def test_same_model_is_same(stored_a):
    v = run(stored_a, "fake:a")
    assert v.status == "same"
    assert v.min_cos >= 0.999
    assert v.reasons == ()


def test_light_noise_is_drift(stored_a):
    v = run(stored_a, "fake:a?noise=0.05")
    assert v.status == "drift"
    assert 0.9 <= v.min_cos < 0.999
    assert "probe_cosine_drift" in v.reasons


def test_other_model_same_dimension_is_different(stored_a):
    v = run(stored_a, "fake:b")
    assert v.status == "different"
    assert {"model_name_changed", "probe_cosine_low"} <= set(v.reasons)
    assert v.min_cos < 0.9


def test_other_dimension_is_different_without_cosine(stored_a):
    v = run(stored_a, "fake:a?dim=256")
    assert v.status == "different"
    assert v.min_cos is None and v.mean_cos is None
    assert v.reasons == ("dim_changed",)


def test_thresholds_change_the_verdict(stored_a):
    strict = Thresholds(same=0.9999999, drift=0.999)
    assert run(stored_a, "fake:a?noise=0.05", thresholds=strict).status == "different"
    loose = Thresholds(same=0.99, drift=0.9)
    assert run(stored_a, "fake:a?noise=0.05", thresholds=loose).status == "same"


def test_prefix_change_is_reported(stored_a):
    v = run(stored_a, "fake:a", doc_prefix="passage: ")
    assert "doc_prefix_changed" in v.reasons
    assert v.status in ("drift", "different")
    v = run(stored_a, "fake:a", query_prefix="query: ")
    assert v.reasons == ("query_prefix_changed",)
    assert v.status == "drift"


def test_compact_fingerprint_still_separates_models():
    compact = stamp(FakeEmbedder("fake:a"), model="fake:a", compact=True)
    assert run(compact, "fake:a").status == "same"
    assert run(compact, "fake:b").status == "different"


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError, match="drift <= same"):
        Thresholds(same=0.5, drift=0.9)


def test_bad_embedder_output_has_a_fix_hint(stored_a):
    with pytest.raises(ValueError, match="one vector per input text"):
        verify(stored_a, lambda texts: [[0.1] * 384], model="fake:a")
    with pytest.raises(ValueError, match=r"\(n, d\)"):
        verify(stored_a, lambda texts: [0.1, 0.2], model="fake:a")

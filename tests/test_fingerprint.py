import json

import pytest

from embedstamp import FakeEmbedder, Fingerprint, resolve_embedder, stamp, verify


def test_json_roundtrip_is_identical(stored_a):
    assert Fingerprint.from_json(stored_a.to_json()) == stored_a


def test_size_limits(stored_a):
    assert len(stored_a.to_json().encode()) <= 12000
    compact = stamp(FakeEmbedder("fake:a"), model="fake:a", compact=True)
    assert compact.probes_fp16 is None
    assert len(compact.to_json().encode()) <= 2500


def test_tiny_noise_keeps_the_digest(stored_a):
    noisy = stamp(FakeEmbedder("fake:a", noise=1e-5), model="fake:a")
    assert noisy.digest == stored_a.digest


def test_different_model_changes_the_digest(stored_a):
    assert stamp(FakeEmbedder("fake:b"), model="fake:b").digest != stored_a.digest


def test_scaled_vectors_keep_cosine_but_report_norm(stored_a):
    embed, model = resolve_embedder("fake:a?scale=3.7")
    v = verify(stored_a, embed, model=model)
    assert v.min_cos == pytest.approx(1.0, abs=1e-6)
    assert "norm_changed" in v.reasons
    assert v.status == "drift"
    assert stored_a.normalized and not v.current.normalized


def test_different_probe_set_is_reported(stored_a):
    probes = ["alpha beta", "gamma delta", "epsilon"]
    v = verify(stored_a, FakeEmbedder("fake:a"), model="fake:a", probes=probes)
    assert "probe_set_changed" in v.reasons
    assert v.status == "different"


def test_custom_probe_set_roundtrip():
    probes = ["alpha beta", "gamma delta", "epsilon"]
    fp = stamp(FakeEmbedder("fake:a"), model="fake:a", probes=probes)
    assert fp.probe_set.startswith("sha256:")
    assert verify(fp, FakeEmbedder("fake:a"), model="fake:a", probes=probes).status == "same"


def test_from_json_rejects_garbage():
    with pytest.raises(ValueError, match="re-create it"):
        Fingerprint.from_json("not json")
    with pytest.raises(ValueError, match="unsupported schema"):
        Fingerprint.from_json(json.dumps({"schema": 99}))


def test_stamp_validates_inputs():
    with pytest.raises(ValueError, match="model"):
        stamp(FakeEmbedder("fake:a"), model="")
    with pytest.raises(ValueError, match="at least 2"):
        stamp(FakeEmbedder("fake:a"), model="m", probes=["only one"])


def test_embedder_with_nan_is_rejected():
    import numpy as np

    with pytest.raises(ValueError, match="NaN"):
        stamp(lambda texts: np.full((len(texts), 4), np.nan), model="m")

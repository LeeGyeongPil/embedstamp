"""Real-model check. Skipped unless EMBEDSTAMP_REAL=1 (downloads ~167 MB on first run).

    EMBEDSTAMP_REAL=1 uv run pytest -q -m real
"""

import os

import pytest

from embedstamp import resolve_embedder, stamp, verify

pytestmark = [
    pytest.mark.real,
    pytest.mark.skipif(os.environ.get("EMBEDSTAMP_REAL") != "1", reason="set EMBEDSTAMP_REAL=1 to run"),
]


def test_real_models_same_and_different():
    bge, name_bge = resolve_embedder("fastembed:BAAI/bge-small-en-v1.5")
    minilm, name_minilm = resolve_embedder("fastembed:sentence-transformers/all-MiniLM-L6-v2")
    fp = stamp(bge, model=name_bge)
    assert verify(fp, bge, model=name_bge).status == "same"
    assert verify(fp, minilm, model=name_minilm).status == "different"

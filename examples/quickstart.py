"""Quickstart: stamp an index, then see same / drift / different and a blocked query.

Runs offline with the practice-mock embedder (no keys, no model download):

    uv run python examples/quickstart.py
"""

import tempfile
from pathlib import Path

from embedstamp import (
    EmbeddingMismatchError,
    FakeEmbedder,
    Guard,
    SidecarStore,
    stamp,
    verify,
)

with tempfile.TemporaryDirectory() as tmp:
    store = SidecarStore(Path(tmp) / "index.embedstamp.json")

    # 1) 인덱스를 만들 때 쓴 모델의 지문을 한 번 기록한다
    embed = FakeEmbedder("fake:a")
    store.save(stamp(embed, model="fake:a"))
    stored = store.load()

    # 2) 나중에 현재 임베더가 그 지문과 맞는지 판정한다
    candidates = {
        "same": FakeEmbedder("fake:a"),                 # 같은 모델을 다시 불러옴
        "drift": FakeEmbedder("fake:a", noise=0.05),    # 양자화처럼 조금 달라진 모델
        "different": FakeEmbedder("fake:b"),            # 같은 384차원의 다른 모델
    }
    for label, candidate in candidates.items():
        v = verify(stored, candidate, model=candidate.model)
        print(f"{label:<10} -> {v.status:<9} min_cos={v.min_cos:.4f} reasons={list(v.reasons)}")

    # 3) Guard 는 질의 전에 확인해서, 다른 모델이면 예외로 막는다
    guard = Guard(store, FakeEmbedder("fake:b"), model="fake:b")
    try:
        guard.embed_query("how do I reset my password?")
    except EmbeddingMismatchError as exc:
        print(f"EmbeddingMismatchError caught: {exc.verdict.status} ({', '.join(exc.verdict.reasons)})")

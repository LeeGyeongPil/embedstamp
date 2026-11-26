"""Offline practice demo: the silent same-dimension model swap, caught and repaired.

Read top to bottom: build an index with model A, swap in model B, watch results change without
an error, let the fingerprint catch it, migrate, switch. Uses the fake embedder only.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import numpy as np

from ._vectors import Embedder
from .embedders import FakeEmbedder
from .fingerprint import stamp
from .i18n import Translator
from .migrate import migrate, overlap_at_k
from .stores.local import LocalIndex
from .verify import verify

K = 10


def load_lines(name: str) -> list[str]:
    """Read a bundled data file (one entry per line)."""
    text = resources.files("embedstamp").joinpath("data", name).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def build_index(path: Path, embed: Embedder, model: str) -> LocalIndex:
    """Index the demo corpus with ``embed`` and stamp it with ``model``'s fingerprint."""
    docs = load_lines("demo_corpus.txt")
    index = LocalIndex(path)
    index.add([f"doc-{i:03d}" for i in range(len(docs))], docs, np.asarray(embed(docs)))
    index.save(stamp(embed, model=model))
    return index


def mean_overlap_at_k(
    index: LocalIndex, queries: list[str], reference: Embedder, current: Embedder, k: int = K
) -> float:
    """Average top-k overlap between results found with ``reference`` and with ``current``."""
    scores = [
        overlap_at_k(
            index.query(np.asarray(reference([q]))[0], k), index.query(np.asarray(current([q]))[0], k), k
        )
        for q in queries
    ]
    return float(np.mean(scores))


def run_demo(tr: Translator, emit: Callable[[str], None] = print) -> int:
    """Run the five-step story, writing lines through ``emit``. Returns an exit code (0)."""
    queries = load_lines("demo_queries.txt")
    model_a, model_b = "fake:a", "fake:b"
    embed_a, embed_b = FakeEmbedder(model_a), FakeEmbedder(model_b)

    emit(tr("mock.banner"))
    with tempfile.TemporaryDirectory(prefix="embedstamp-demo-") as tmp:
        # 1) 모델 A 로 인덱스를 만들고 지문을 찍는다
        emit(tr("demo.step1", n=len(load_lines("demo_corpus.txt")), model=model_a, dim=embed_a.dim))
        index = build_index(Path(tmp) / "demo", embed_a, model_a)
        fp = index.load()
        assert fp is not None
        emit(tr("demo.stamped", model=fp.model, dim=fp.dim, digest=fp.digest[:12]))

        # 2) 같은 차원의 모델 B 로 바꿔 질의한다 - 예외가 없고 결과만 조용히 바뀐다
        emit(tr("demo.step2", model=model_b, dim=embed_b.dim, n=len(queries)))
        baseline = mean_overlap_at_k(index, queries, embed_a, embed_a)
        swapped = mean_overlap_at_k(index, queries, embed_a, embed_b)
        emit(tr("demo.noerror"))
        emit(tr("demo.overlap", k=K, value=f"{swapped:.2f}", baseline=f"{baseline:.2f}"))

        # 3) 지문으로 불일치를 판정한다
        emit(tr("demo.step3"))
        verdict = verify(fp, embed_b, model=model_b)
        emit("    " + tr("check.status", status=verdict.status, min_cos=f"{verdict.min_cos:.2f}",
                         mean_cos=f"{verdict.mean_cos:.2f}"))
        emit("    " + tr("check.reasons") + " " + ", ".join(verdict.reasons))
        emit(tr("demo.guard"))

        # 4) green 컬렉션에 새 모델로 다시 임베딩한다
        emit(tr("demo.step4", model=model_b))
        report = migrate(index, embed_b, new_model=model_b, old_embed=embed_a, queries=queries, k=K)
        emit("    " + tr("migrate.report", n_docs=report.n_docs, n_embedded=report.n_embedded,
                         n_skipped=report.n_skipped, green=report.green_name))
        emit("    " + tr("migrate.self_recall", k=K, value=f"{report.self_recall_at_k:.2f}"))
        emit("    " + tr("migrate.overlap", k=K, value=f"{report.overlap_at_k:.2f}"))

        # 5) 이름을 전환하고 새 지문이 일치하는지 다시 확인한다
        emit(tr("demo.step5"))
        report = migrate(index, embed_b, new_model=model_b, switch=True)
        emit("    " + tr("migrate.switched", store=index.name, model=model_b, blue=report.blue_name))
        live = index.load()
        assert live is not None
        after = verify(live, embed_b, model=model_b)
        emit("    " + tr("check.status", status=after.status, min_cos=f"{after.min_cos:.2f}",
                         mean_cos=f"{after.mean_cos:.2f}"))
    emit(tr("demo.done"))
    return 0

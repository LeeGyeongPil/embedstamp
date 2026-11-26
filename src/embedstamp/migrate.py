"""Blue/green re-embedding: build a new collection with the new model, measure it, then switch."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ._vectors import Embedder, as_matrix
from .errors import EmbedstampError, SwitchRefusedError
from .fingerprint import Fingerprint, stamp
from .stores import Doc, MigratableStore
from .verify import Verdict, verify

_SAMPLE_LIMIT = 100

EMPTY_SOURCE_MESSAGE = (
    "the source index is empty; there is nothing to migrate. migrate() copies the documents of an "
    "existing index, so add them first (LocalIndex.add(...) or your vector store's add) and run it again"
)


@dataclass(frozen=True)
class MigrationReport:
    """What migrate() did and how good the new collection is."""

    n_docs: int
    n_embedded: int
    n_skipped: int
    k: int
    self_recall_at_k: float
    overlap_at_k: float | None
    green_name: str
    blue_name: str | None
    switched: bool
    status_after_switch: str | None
    fingerprint: Fingerprint
    seconds: float
    # 전환 직후 다시 잰 판정(숫자 포함). 전환하지 않았으면 None.
    verdict_after_switch: Verdict | None = None
    # green 컬렉션을 만든 이름(예: docs__green). 전환하면 green_name 은 원래 이름(docs)이 된다.
    built_name: str = ""


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """Share of the top-``k`` ids that two result lists have in common (0..1)."""
    return len(set(a[:k]) & set(b[:k])) / k


def migrate(
    source: MigratableStore,
    new_embed: Embedder,
    *,
    new_model: str,
    target_name: str | None = None,
    old_embed: Embedder | None = None,
    queries: Sequence[str] | None = None,
    k: int = 10,
    batch_size: int = 64,
    switch: bool = False,
    doc_prefix: str = "",
    query_prefix: str = "",
    old_query_prefix: str = "",
    min_self_recall: float = 0.9,
    compact: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> MigrationReport:
    """Re-embed every document of ``source`` into a new collection with ``new_embed``.

    1. Create (or resume) the green collection ``target_name`` (default ``<name>__green``).
    2. Embed documents in batches; ids already present in green are skipped, so a re-run after
       a crash continues where it stopped.
    3. Measure self-recall@k (documents found by their own text) and, when ``old_embed`` is
       given, the top-k overlap between old and new results for ``queries``.
    4. Only with ``switch=True`` (and self-recall >= ``min_self_recall``): rename so that green
       takes the original name; the old data stays as ``<name>__blue_<date>``.
    """
    if k < 1 or batch_size < 1:
        raise ValueError("k and batch_size must be >= 1")
    started = time.monotonic()
    n_docs = source.count()
    if n_docs == 0:
        raise EmbedstampError(EMPTY_SOURCE_MESSAGE)

    green = source.create_sibling(target_name or f"{source.name}__green")
    existing_fp = green.load()
    if existing_fp is not None:
        verdict = verify(existing_fp, new_embed, model=new_model, doc_prefix=doc_prefix,
                         query_prefix=query_prefix)
        if verdict.status == "different":
            raise EmbedstampError(
                f"collection {green.name!r} already holds vectors from a different model "
                f"({existing_fp.model!r}); delete it or choose another target_name"
            )
    elif green.count() > 0:
        raise EmbedstampError(
            f"collection {green.name!r} is not empty and has no fingerprint, so its model is "
            "unknown; delete it or choose another target_name"
        )
    else:
        green.save(stamp(new_embed, model=new_model, doc_prefix=doc_prefix,
                         query_prefix=query_prefix, compact=compact))

    done = green.existing_ids()
    n_embedded = n_skipped = 0
    stride = max(1, n_docs // _SAMPLE_LIMIT)
    sample: list[Doc] = []
    seen = 0
    for batch in source.iter_documents(batch_size):
        todo = [d for d in batch if d.id not in done]
        n_skipped += len(batch) - len(todo)
        if todo:
            vectors = as_matrix(new_embed([doc_prefix + d.text for d in todo]), len(todo))
            green.add([d.id for d in todo], [d.text for d in todo], vectors, [d.metadata for d in todo])
            n_embedded += len(todo)
        for d in batch:
            if seen % stride == 0 and len(sample) < _SAMPLE_LIMIT:
                sample.append(d)
            seen += 1
        if progress:
            progress(n_embedded + n_skipped, n_docs)

    probe_texts = [query_prefix + d.text for d in sample]
    sample_vecs = as_matrix(new_embed(probe_texts), len(probe_texts))
    hits = sum(d.id in green.query(v, k) for d, v in zip(sample, sample_vecs))
    self_recall = hits / len(sample)

    overlap: float | None = None
    if old_embed is not None:
        qtexts = list(queries) if queries else [d.text for d in sample]
        old_vecs = as_matrix(old_embed([old_query_prefix + q for q in qtexts]), len(qtexts))
        new_vecs = as_matrix(new_embed([query_prefix + q for q in qtexts]), len(qtexts))
        overlap = float(np.mean([
            overlap_at_k(source.query(o, k), green.query(n, k), k)
            for o, n in zip(old_vecs, new_vecs)
        ]))

    new_fp = green.load()
    assert new_fp is not None  # stamped above or on a previous run
    blue_label: str | None = None
    switched = False
    status_after: str | None = None
    green_label = green.name
    report = MigrationReport(
        n_docs=n_docs, n_embedded=n_embedded, n_skipped=n_skipped, k=k,
        self_recall_at_k=self_recall, overlap_at_k=overlap, green_name=green_label,
        blue_name=None, switched=False, status_after_switch=None, fingerprint=new_fp,
        seconds=time.monotonic() - started, built_name=green_label,
    )
    if switch:
        if self_recall < min_self_recall:
            raise SwitchRefusedError(
                f"self-recall@{k} is {self_recall:.2f}, below the required {min_self_recall:.2f}; "
                f"collection {green_label!r} was kept but not switched. Check the embedder and prefixes.",
                report,
            )
        blue_label = source.switch_to(green)
        switched = True
        # 전환 뒤 원래 이름에 실린 지문을 새 임베더로 다시 재서, 상태와 수치를 함께 보고한다
        live_fp = source.load()
        verdict_after: Verdict | None = None
        if live_fp is not None:
            verdict_after = verify(live_fp, new_embed, model=new_model, doc_prefix=doc_prefix,
                                   query_prefix=query_prefix)
            status_after = verdict_after.status
        report = MigrationReport(
            n_docs=n_docs, n_embedded=n_embedded, n_skipped=n_skipped, k=k,
            self_recall_at_k=self_recall, overlap_at_k=overlap, green_name=source.name,
            blue_name=blue_label, switched=switched, status_after_switch=status_after,
            fingerprint=new_fp, seconds=time.monotonic() - started,
            verdict_after_switch=verdict_after, built_name=green_label,
        )
    return report

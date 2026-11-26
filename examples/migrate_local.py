"""Blue/green migration: re-embed a LocalIndex with a new model, check quality, then switch.

Runs offline with the practice-mock embedders:

    uv run python examples/migrate_local.py
"""

import tempfile
from pathlib import Path

from embedstamp import FakeEmbedder, LocalIndex, Guard, migrate, stamp

docs = [f"document number {i} about topic {i % 7}" for i in range(40)]

with tempfile.TemporaryDirectory() as tmp:
    index = LocalIndex(Path(tmp) / "kb")
    old = FakeEmbedder("fake:a")
    index.add([f"d{i}" for i in range(len(docs))], docs, old(docs))
    index.save(stamp(old, model="fake:a"))  # stamp the index while it still matches its model

    new = FakeEmbedder("fake:b")  # same 384 dimensions, different model
    report = migrate(index, new, new_model="fake:b", old_embed=old, queries=docs[:10], switch=True)
    print(f"embedded={report.n_embedded} skipped={report.n_skipped} self_recall@{report.k}={report.self_recall_at_k:.2f}")
    print(f"switched={report.switched} status_after_switch={report.status_after_switch}")

    live = LocalIndex(Path(tmp) / "kb")  # the original name now serves fake:b
    print("live model:", live.load().model)
    print("query ok:", Guard(live, new, model="fake:b").embed_query("topic 3").shape)

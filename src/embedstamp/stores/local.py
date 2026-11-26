"""LocalIndex: a tiny brute-force cosine index kept in a directory (docs, vectors, fingerprint).

Meant for demos, tests and small corpora. Layout: ``docs.jsonl``, ``vectors.npy``,
``embedstamp.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._vectors import unit_rows
from ..fingerprint import Fingerprint
from . import Doc, blue_name
from .sidecar import atomic_write


class LocalIndex:
    """Directory-backed vector index with cosine search."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._vecs = np.zeros((0, 0), dtype=np.float32)
        self._reload()

    @property
    def name(self) -> str:
        return self.path.name

    # ---- fingerprint -------------------------------------------------
    def load(self) -> Fingerprint | None:
        file = self.path / "embedstamp.json"
        if not file.exists():
            return None
        return Fingerprint.from_file(file)

    def save(self, fp: Fingerprint) -> None:
        atomic_write(self.path / "embedstamp.json", fp.to_json())

    # ---- persistence -------------------------------------------------
    def _reload(self) -> None:
        docs, vecs = self.path / "docs.jsonl", self.path / "vectors.npy"
        self._ids, self._texts, self._metas = [], [], []
        self._vecs = np.zeros((0, 0), dtype=np.float32)
        if not (docs.exists() and vecs.exists()):
            return
        rows = [json.loads(line) for line in docs.read_text(encoding="utf-8").splitlines() if line]
        matrix = np.load(vecs)
        n = min(len(rows), matrix.shape[0])  # tolerate a crash between the two writes
        self._ids = [r["id"] for r in rows[:n]]
        self._texts = [r["text"] for r in rows[:n]]
        self._metas = [r.get("metadata", {}) for r in rows[:n]]
        self._vecs = matrix[:n].astype(np.float32)

    def _persist(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.path / "vectors.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, self._vecs)
        os.replace(tmp, self.path / "vectors.npy")
        lines = [
            json.dumps({"id": i, "text": t, "metadata": m}, ensure_ascii=False)
            for i, t, m in zip(self._ids, self._texts, self._metas)
        ]
        atomic_write(self.path / "docs.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    # ---- index API ---------------------------------------------------
    def count(self) -> int:
        return len(self._ids)

    def existing_ids(self) -> set[str]:
        return set(self._ids)

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or replace documents together with their vectors."""
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or not (len(ids) == len(texts) == vecs.shape[0]):
            raise ValueError("ids, texts and vectors must have the same length; vectors must be 2-D")
        if self._vecs.shape[0] and vecs.shape[1] != self._vecs.shape[1]:
            raise ValueError(
                f"vector dimension {vecs.shape[1]} does not match the index dimension "
                f"{self._vecs.shape[1]}; the index was built with a different embedding model"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique within one add() call")
        metas = list(metadatas) if metadatas is not None else [{} for _ in ids]
        position = {doc_id: i for i, doc_id in enumerate(self._ids)}
        fresh_rows: list[np.ndarray] = []
        for doc_id, text, meta, vec in zip(ids, texts, metas, vecs):
            if doc_id in position:
                i = position[doc_id]
                self._texts[i], self._metas[i], self._vecs[i] = text, dict(meta), vec
            else:
                position[doc_id] = len(self._ids) + len(fresh_rows)
                self._ids.append(doc_id)
                self._texts.append(text)
                self._metas.append(dict(meta))
                fresh_rows.append(vec)
        if fresh_rows:
            stack = np.stack(fresh_rows)
            self._vecs = stack if self._vecs.shape[0] == 0 else np.vstack([self._vecs, stack])
        self._persist()

    def query(self, vector: np.ndarray, k: int = 10) -> list[str]:
        """Ids of the ``k`` most cosine-similar documents. Wrong dimensions raise ValueError."""
        if not self._ids:
            return []
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._vecs.shape[1]:
            raise ValueError(
                f"query dimension {vec.shape[1]} does not match the index dimension {self._vecs.shape[1]}"
            )
        scores = (unit_rows(self._vecs) @ unit_rows(vec).T).ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._ids[i] for i in order]

    def iter_documents(self, batch_size: int = 64) -> Iterator[list[Doc]]:
        for start in range(0, len(self._ids), batch_size):
            end = start + batch_size
            yield [
                Doc(i, t, dict(m))
                for i, t, m in zip(self._ids[start:end], self._texts[start:end], self._metas[start:end])
            ]

    # ---- blue/green --------------------------------------------------
    def create_sibling(self, name: str) -> "LocalIndex":
        """Open (or start) an empty index next to this one; an existing one is reused to resume."""
        return LocalIndex(self.path.with_name(name))

    def switch_to(self, sibling: "LocalIndex") -> str:
        """Keep this index as ``<name>__blue_<date>`` and make ``sibling``'s data live under the
        original name. Returns the blue name."""
        taken = {p.name for p in self.path.parent.iterdir()}
        blue_label = blue_name(self.name, taken)
        blue = self.path.with_name(blue_label)
        os.replace(self.path, blue)
        try:
            os.replace(sibling.path, self.path)
        except OSError as exc:
            os.replace(blue, self.path)  # roll back so the live index is untouched
            raise OSError(f"switch failed and was rolled back: {exc}") from exc
        sibling.path = self.path
        self._reload()
        return blue_label

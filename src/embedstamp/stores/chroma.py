"""ChromaStore: keep the fingerprint in a Chroma collection's metadata (extra ``chroma``).

Notes verified against chromadb 1.5.9: ``collection.modify(metadata=...)`` *replaces* the whole
metadata dict and refuses to receive ``hnsw:*`` keys again (the distance function lives in the
collection configuration and is kept). So ``save`` merges the existing non-``hnsw`` keys itself.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..errors import FingerprintTooLargeError, missing_extra_error
from ..fingerprint import Fingerprint
from . import Doc, blue_name

METADATA_KEY = "embedstamp"


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise missing_extra_error("ChromaStore", "chromadb", "chroma") from exc
    return chromadb


class ChromaStore:
    """Adapter for a ``chromadb`` collection (vectors added directly, ``embedding_function=None``)."""

    def __init__(self, collection: Any, client: Any | None = None) -> None:
        self._collection = collection
        self._client = client if client is not None else getattr(collection, "_client", None)

    @classmethod
    def open(cls, db_dir: str, name: str) -> "ChromaStore":
        """Open an existing collection of a persistent Chroma database directory."""
        chromadb = _import_chromadb()
        from chromadb.config import Settings

        # PersistentClient would silently create the directory; reads must not write.
        if not os.path.isdir(db_dir):
            raise ValueError(f"Chroma database directory not found: {db_dir}")

        client = chromadb.PersistentClient(path=db_dir, settings=Settings(anonymized_telemetry=False))
        try:
            collection = client.get_collection(name, embedding_function=None)
        except Exception as exc:
            raise ValueError(f"collection {name!r} not found in {db_dir!r}: {exc}") from exc
        return cls(collection, client)

    @property
    def name(self) -> str:
        return str(self._collection.name)

    @property
    def collection(self) -> Any:
        return self._collection

    # ---- fingerprint -------------------------------------------------
    def load(self) -> Fingerprint | None:
        raw = (self._collection.metadata or {}).get(METADATA_KEY)
        if raw is None:
            return None
        return Fingerprint.from_json(str(raw))

    def save(self, fp: Fingerprint) -> None:
        merged = {
            k: v for k, v in (self._collection.metadata or {}).items() if not k.startswith("hnsw:")
        }
        merged[METADATA_KEY] = fp.to_json()
        try:
            self._collection.modify(metadata=merged)
        except Exception as exc:
            if any(word in str(exc).lower() for word in ("large", "size", "length", "limit")):
                raise FingerprintTooLargeError(
                    f"Chroma refused the fingerprint metadata ({exc}); "
                    "use stamp(..., compact=True) or a SidecarStore"
                ) from exc
            raise

    # ---- index API ---------------------------------------------------
    def count(self) -> int:
        return int(self._collection.count())

    def existing_ids(self) -> set[str]:
        return set(self._collection.get(include=[])["ids"])

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents with precomputed vectors (safe to repeat when resuming)."""
        metas: list[dict[str, Any] | None] | None = None
        if metadatas is not None and any(metadatas):
            metas = [dict(m) if m else None for m in metadatas]
        self._collection.upsert(
            ids=list(ids),
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            documents=list(texts),
            metadatas=metas,
        )

    def query(self, vector: np.ndarray, k: int = 10) -> list[str]:
        n = self.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()], n_results=min(k, n)
        )
        return list(res["ids"][0])

    def iter_documents(self, batch_size: int = 64) -> Iterator[list[Doc]]:
        offset = 0
        while True:
            res = self._collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            ids = res["ids"]
            if not ids:
                return
            docs = res["documents"] or [""] * len(ids)
            metas = res["metadatas"] or [None] * len(ids)
            yield [Doc(i, d or "", dict(m or {})) for i, d, m in zip(ids, docs, metas)]
            offset += len(ids)

    # ---- blue/green --------------------------------------------------
    def create_sibling(self, name: str) -> "ChromaStore":
        """Get or create an empty collection with the same distance function and extra metadata."""
        meta = {
            k: v
            for k, v in (self._collection.metadata or {}).items()
            if not k.startswith("hnsw:") and k != METADATA_KEY
        }
        kwargs: dict[str, Any] = {"metadata": meta or None, "embedding_function": None}
        try:
            space = self._collection.configuration["hnsw"]["space"]
            kwargs["configuration"] = {"hnsw": {"space": space}}
        except (KeyError, TypeError, AttributeError):
            pass
        return ChromaStore(self._client.get_or_create_collection(name, **kwargs), self._client)

    def switch_to(self, sibling: "ChromaStore") -> str:
        """Rename: this collection becomes ``<name>__blue_<date>``, ``sibling`` takes the old name.

        Chroma has no aliases, so this is two renames and not atomic. If the second rename
        fails the first is undone.
        """
        original = self.name
        taken = {c.name for c in self._client.list_collections()}
        blue_label = blue_name(original, taken)
        self._collection.modify(name=blue_label)
        try:
            sibling._collection.modify(name=original)
        except Exception as exc:
            self._collection.modify(name=original)
            raise RuntimeError(f"switch failed and was rolled back: {exc}") from exc
        self._collection = sibling._collection
        return blue_label

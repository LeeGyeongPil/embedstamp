"""Storage adapters: where fingerprints live, and what migrate() needs from an index."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..fingerprint import Fingerprint


@dataclass(frozen=True)
class Doc:
    """One stored document."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FingerprintStore(Protocol):
    """Anything that can keep one fingerprint."""

    def load(self) -> Fingerprint | None: ...

    def save(self, fp: Fingerprint) -> None: ...


@runtime_checkable
class MigratableStore(FingerprintStore, Protocol):
    """An index that migrate() can copy into a new collection and switch."""

    name: str

    def iter_documents(self, batch_size: int = 64) -> Iterator[list[Doc]]: ...

    def create_sibling(self, name: str) -> "MigratableStore": ...

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None: ...

    def query(self, vector: np.ndarray, k: int = 10) -> list[str]: ...

    def existing_ids(self) -> set[str]: ...

    def count(self) -> int: ...

    def switch_to(self, sibling: "MigratableStore") -> str:
        """Keep this index under a ``__blue_<date>`` name, make ``sibling`` live; return the blue name."""
        ...


def blue_name(name: str, taken: set[str] | None = None) -> str:
    """Name under which the previous live collection is kept: ``<name>__blue_<YYYYMMDD>``."""
    base = f"{name}__blue_{datetime.now(timezone.utc):%Y%m%d}"
    candidate, n = base, 2
    while taken and candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def resolve_store(spec: str) -> FingerprintStore:
    """Open a store from a spec: ``local://<dir>``, ``sidecar://<file>`` or ``chroma://<db_dir>/<collection>``."""
    scheme, sep, rest = spec.partition("://")
    if not sep or not rest:
        raise ValueError(
            f"invalid store spec {spec!r}; use local://<dir>, sidecar://<file> or chroma://<db_dir>/<collection>"
        )
    if scheme == "local":
        from .local import LocalIndex

        return LocalIndex(rest)
    if scheme == "sidecar":
        from .sidecar import SidecarStore

        return SidecarStore(rest)
    if scheme == "chroma":
        from .chroma import ChromaStore

        db_dir, slash, collection = rest.rpartition("/")
        if not slash or not db_dir or not collection:
            raise ValueError("chroma spec must look like chroma://<db_dir>/<collection>")
        return ChromaStore.open(db_dir, collection)
    raise ValueError(f"unknown store scheme {scheme!r}; use local, sidecar or chroma")

"""Guard: check the embedder against the stored fingerprint once, then embed safely."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ._vectors import Embedder, as_matrix
from .errors import EmbeddingMismatchError, FingerprintNotFoundError
from .stores import FingerprintStore
from .verify import Thresholds, Verdict, verify


class Guard:
    """Wrap an embedder so that queries and inserts first prove it matches the index.

    The check embeds the probe sentences once and caches the verdict, so later calls cost
    nothing. ``on_mismatch="raise"`` raises :class:`EmbeddingMismatchError` for a different
    model; ``"warn"`` only emits a :class:`UserWarning`. A ``drift`` verdict warns when
    ``allow_drift`` is true and raises otherwise.
    """

    def __init__(
        self,
        store: FingerprintStore,
        embed: Embedder,
        *,
        model: str,
        doc_prefix: str = "",
        query_prefix: str = "",
        on_mismatch: Literal["raise", "warn"] = "raise",
        allow_drift: bool = True,
        thresholds: Thresholds = Thresholds(),
    ) -> None:
        if on_mismatch not in ("raise", "warn"):
            raise ValueError("on_mismatch must be 'raise' or 'warn'")
        self._store = store
        self._embed = embed
        self._model = model
        self._doc_prefix = doc_prefix
        self._query_prefix = query_prefix
        self._on_mismatch = on_mismatch
        self._allow_drift = allow_drift
        self._thresholds = thresholds
        self._verdict: Verdict | None = None

    def check(self) -> Verdict:
        """Return the verdict (computed on the first call, cached afterwards)."""
        if self._verdict is not None:
            return self._verdict
        stored = self._store.load()
        if stored is None:
            raise FingerprintNotFoundError(type(self._store).__name__)
        verdict = verify(
            stored, self._embed, model=self._model, doc_prefix=self._doc_prefix,
            query_prefix=self._query_prefix, thresholds=self._thresholds,
        )
        blocked = verdict.status == "different" or (verdict.status == "drift" and not self._allow_drift)
        if blocked and self._on_mismatch == "raise":
            raise EmbeddingMismatchError(verdict)  # not cached: the next call checks again
        if verdict.status != "same":
            warnings.warn(
                f"embedstamp: index fingerprint check says {verdict.status} "
                f"(min_cos={verdict.min_cos}, reasons={', '.join(verdict.reasons)})",
                UserWarning,
                stacklevel=3,
            )
        self._verdict = verdict
        return verdict

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Check, then embed ``texts`` with the document prefix; returns ``(n, d)``."""
        self.check()
        batch = [self._doc_prefix + t for t in texts]
        return as_matrix(self._embed(batch), len(batch))

    def embed_query(self, text: str) -> np.ndarray:
        """Check, then embed one query with the query prefix; returns a 1-D vector."""
        self.check()
        return as_matrix(self._embed([self._query_prefix + text]), 1)[0]

"""Small numpy helpers shared by the fingerprint, verify and store modules (internal)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

#: An embedder maps a list of texts to an ``(n, d)`` array-like of vectors.
Embedder = Callable[[Sequence[str]], Any]


def as_matrix(raw: Any, n_expected: int | None = None) -> np.ndarray:
    """Convert an embedder result to a finite float32 ``(n, d)`` matrix or raise ValueError."""
    arr = np.asarray(raw)
    if arr.ndim == 1 and n_expected == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            f"embedder must return an (n, d) array of vectors, got shape {arr.shape}; "
            "wrap your model so that it maps list[str] to a 2-D array"
        )
    if n_expected is not None and arr.shape[0] != n_expected:
        raise ValueError(
            f"embedder returned {arr.shape[0]} vectors for {n_expected} texts; "
            "it must return one vector per input text, in order"
        )
    out = arr.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("embedder returned NaN or infinite values; check the model output")
    return out


def raw_dtype(raw: Any) -> str:
    """Name of the dtype the embedder produced before conversion to float32."""
    return np.asarray(raw).dtype.name


def unit_rows(mat: np.ndarray) -> np.ndarray:
    """Return rows scaled to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of matching rows of ``a`` and ``b``."""
    return np.einsum("ij,ij->i", unit_rows(a.astype(np.float64)), unit_rows(b.astype(np.float64)))

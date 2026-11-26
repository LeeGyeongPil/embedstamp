"""Compare the current embedder with a stored fingerprint: same, drift or different."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._vectors import Embedder, row_cosines
from .fingerprint import Fingerprint, build_fingerprint, measure, project
from .probes import probe_set_id

Status = Literal["same", "drift", "different"]
_RANK = {"same": 0, "drift": 1, "different": 2}


@dataclass(frozen=True)
class Thresholds:
    """Cut-offs for the minimum probe cosine and the allowed relative norm change."""

    same: float = 0.999
    drift: float = 0.9
    norm_rel_tol: float = 0.01

    def __post_init__(self) -> None:
        if not (0.0 <= self.drift <= self.same <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= drift <= same <= 1")
        if self.norm_rel_tol < 0:
            raise ValueError("norm_rel_tol must be >= 0")


@dataclass(frozen=True)
class Verdict:
    """Result of :func:`verify`.

    ``reasons`` holds codes such as ``dim_changed``, ``model_name_changed``,
    ``doc_prefix_changed``, ``query_prefix_changed``, ``norm_changed``,
    ``normalization_changed``, ``probe_cosine_low``, ``probe_cosine_drift`` and
    ``probe_set_changed``.
    """

    status: Status
    min_cos: float | None
    mean_cos: float | None
    reasons: tuple[str, ...]
    stored: Fingerprint
    current: Fingerprint

    @property
    def compatible(self) -> bool:
        """True unless the model is different (drift still counts as compatible)."""
        return self.status != "different"


def verify(
    stored: Fingerprint,
    embed: Embedder,
    *,
    model: str,
    doc_prefix: str = "",
    query_prefix: str = "",
    thresholds: Thresholds = Thresholds(),
    probes: Sequence[str] | None = None,
) -> Verdict:
    """Embed the probe sentences with ``embed`` and judge them against ``stored``.

    Order of checks: static fields (dimension, probe set, names, prefixes, norms), then the
    cosine between stored and fresh probe vectors (minimum over probes).
    """
    vectors, dtype = measure(embed, probes, doc_prefix)
    current = build_fingerprint(
        vectors, model=model, dtype=dtype, doc_prefix=doc_prefix, query_prefix=query_prefix,
        probe_set=probe_set_id(probes), compact=stored.probes_fp16 is None,
    )
    reasons: list[str] = []
    status: Status = "same"

    def raise_to(level: Status) -> None:
        nonlocal status
        if _RANK[level] > _RANK[status]:
            status = level

    if current.dim != stored.dim:
        return Verdict("different", None, None, ("dim_changed",), stored, current)
    if current.probe_set != stored.probe_set or vectors.shape[0] != stored.n_probes:
        return Verdict("different", None, None, ("probe_set_changed",), stored, current)

    if current.model != stored.model:
        reasons.append("model_name_changed")
        raise_to("drift")
    if current.doc_prefix != stored.doc_prefix:
        reasons.append("doc_prefix_changed")
        raise_to("drift")
    if current.query_prefix != stored.query_prefix:
        reasons.append("query_prefix_changed")
        raise_to("drift")
    if abs(current.norm_mean - stored.norm_mean) > thresholds.norm_rel_tol * max(stored.norm_mean, 1e-9):
        reasons.append("norm_changed")
        raise_to("drift")
    if current.normalized != stored.normalized:
        reasons.append("normalization_changed")
        raise_to("drift")

    stored_vectors = stored.probe_vectors()
    if stored_vectors is not None:
        cos = row_cosines(stored_vectors, vectors)
    else:  # compact fingerprint: compare the 32-dimensional projections
        cos = row_cosines(stored.projection_matrix(), project(vectors))
    min_cos = float(np.min(cos))
    mean_cos = float(np.mean(cos))
    if min_cos < thresholds.drift:
        reasons.append("probe_cosine_low")
        raise_to("different")
    elif min_cos < thresholds.same:
        reasons.append("probe_cosine_drift")
        raise_to("drift")
    return Verdict(status, min_cos, mean_cos, tuple(reasons), stored, current)

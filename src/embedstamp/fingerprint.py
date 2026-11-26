"""The fingerprint: what gets stored next to a vector index to identify its embedding model."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np

from ._vectors import Embedder, as_matrix, raw_dtype, unit_rows
from .probes import probe_set_id, resolve_probes

SCHEMA_VERSION = 1
PROJECTION_DIMS = 32
_PROJECTION_SEED = 20261127


@lru_cache(maxsize=8)
def _projection_matrix(dim: int) -> np.ndarray:
    """Seeded random projection ``dim -> 32``; identical on every machine for a given dim."""
    rng = np.random.default_rng(_PROJECTION_SEED + dim)
    return (rng.standard_normal((dim, PROJECTION_DIMS)) / np.sqrt(PROJECTION_DIMS)).astype(np.float64)


def project(vectors: np.ndarray) -> np.ndarray:
    """Project unit-length probe vectors to 32 dimensions, shape ``(n, 32)``."""
    return unit_rows(vectors.astype(np.float64)) @ _projection_matrix(vectors.shape[1])


@dataclass(frozen=True)
class Fingerprint:
    """Identity card of the embedding model that produced the vectors in an index."""

    schema: int
    model: str
    dim: int
    dtype: str
    normalized: bool
    norm_mean: float
    doc_prefix: str
    query_prefix: str
    probe_set: str
    probes_fp16: str | None
    projection32: list[float]
    digest: str
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_probes(self) -> int:
        return len(self.projection32) // PROJECTION_DIMS

    def probe_vectors(self) -> np.ndarray | None:
        """Decode the stored probe vectors (float16 precision), or None for compact fingerprints."""
        if self.probes_fp16 is None:
            return None
        raw = base64.b64decode(self.probes_fp16)
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(self.n_probes, self.dim)

    def projection_matrix(self) -> np.ndarray:
        return np.asarray(self.projection32, dtype=np.float64).reshape(self.n_probes, PROJECTION_DIMS)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Fingerprint":
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            if data.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema {data.get('schema')!r}")
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid embedstamp fingerprint ({exc}); re-create it with embedstamp.stamp()"
            ) from exc

    @classmethod
    def from_file(cls, path: "str | os.PathLike[str]") -> "Fingerprint":
        """Read a fingerprint file; undecodable bytes get the same fix-it message as bad JSON."""
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"invalid embedstamp fingerprint ({exc}); re-create it with embedstamp.stamp()"
            ) from exc
        return cls.from_json(text)


def _digest(
    model: str, dim: int, dtype: str, doc_prefix: str, query_prefix: str, probe_set: str,
    projection: np.ndarray,
) -> str:
    """Hash of the static fields plus the sign pattern of the projected probes.

    Signs (a SimHash) make the digest immune to tiny numeric noise, while a different
    model flips about half of the bits.
    """
    bits = np.packbits((projection >= 0).astype(np.uint8)).tobytes()
    head = "\x1f".join([model, str(dim), dtype, doc_prefix, query_prefix, probe_set])
    return hashlib.sha256(head.encode("utf-8") + b"\x00" + bits).hexdigest()[:32]


def build_fingerprint(
    vectors: np.ndarray,
    *,
    model: str,
    dtype: str,
    doc_prefix: str = "",
    query_prefix: str = "",
    probe_set: str,
    compact: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> Fingerprint:
    """Build a fingerprint from already embedded probe vectors."""
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    proj = project(vectors)
    proj_stored = np.round(proj, 3)
    probes_fp16 = None
    if not compact:
        probes_fp16 = base64.b64encode(vectors.astype("<f2").tobytes()).decode("ascii")
    return Fingerprint(
        schema=SCHEMA_VERSION,
        model=model,
        dim=int(vectors.shape[1]),
        dtype=dtype,
        normalized=bool(np.all(np.abs(norms - 1.0) <= 1e-3)),
        norm_mean=round(float(norms.mean()), 4),
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
        probe_set=probe_set,
        probes_fp16=probes_fp16,
        projection32=[float(x) for x in proj_stored.reshape(-1)],
        digest=_digest(model, int(vectors.shape[1]), dtype, doc_prefix, query_prefix, probe_set, proj_stored),
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        extra=dict(extra or {}),
    )


def measure(embed: Embedder, probes: Sequence[str] | None, doc_prefix: str) -> tuple[np.ndarray, str]:
    """Embed the probe sentences (with the document prefix) and return ``(vectors, dtype)``."""
    texts = [doc_prefix + p for p in resolve_probes(probes)]
    raw = embed(texts)
    return as_matrix(raw, len(texts)), raw_dtype(raw)


def stamp(
    embed: Embedder,
    *,
    model: str,
    doc_prefix: str = "",
    query_prefix: str = "",
    probes: Sequence[str] | None = None,
    compact: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> Fingerprint:
    """Embed the probe sentences with ``embed`` and return the fingerprint to store.

    ``model`` is the name you want to record (for example ``"BAAI/bge-small-en-v1.5"``).
    ``compact=True`` keeps only a 32-dimensional summary (about 2 KB) for stores with small
    metadata limits; comparisons are then less precise.
    """
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string, e.g. model='BAAI/bge-small-en-v1.5'")
    vectors, dtype = measure(embed, probes, doc_prefix)
    return build_fingerprint(
        vectors, model=model, dtype=dtype, doc_prefix=doc_prefix, query_prefix=query_prefix,
        probe_set=probe_set_id(probes), compact=compact, extra=extra,
    )

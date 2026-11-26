"""Fixed probe sentences. Embedding the same sentences again reveals whether the model changed."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

DEFAULT_PROBE_SET = "default-v1"

# Mixed English, Korean, numbers and code so that tokenizer or language-coverage changes show up.
DEFAULT_PROBES: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "How do I reset my password?",
    "서울에서 부산까지 KTX로 약 두 시간 반이 걸린다.",
    "Invoice #4821 is due on 2026-03-31 for $1,250.00.",
    "def add(a, b): return a + b",
    "Photosynthesis converts light energy into chemical energy.",
    "오늘 회의는 오후 세 시에 시작합니다.",
    "A vector index stores embeddings for similarity search.",
)


def probe_set_id(probes: Sequence[str] | None) -> str:
    """Return the id of a probe set: ``default-v1`` or ``sha256:<12 hex>`` for custom sets."""
    if probes is None or tuple(probes) == DEFAULT_PROBES:
        return DEFAULT_PROBE_SET
    digest = hashlib.sha256("\x1f".join(probes).encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def resolve_probes(probes: Sequence[str] | None) -> tuple[str, ...]:
    """Return the probe sentences to use, validating custom sets."""
    if probes is None:
        return DEFAULT_PROBES
    items = tuple(probes)
    if len(items) < 2 or any(not isinstance(p, str) or not p.strip() for p in items):
        raise ValueError("probes must be at least 2 non-empty strings")
    return items

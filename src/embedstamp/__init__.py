"""embedstamp: stamp a vector index with its embedding model's fingerprint.

Quick tour::

    fp = stamp(embed, model="my-model")        # fingerprint from fixed probe sentences
    SidecarStore("index.embedstamp.json").save(fp)
    verdict = verify(fp, embed, model="my-model")   # same / drift / different
    guard = Guard(store, embed, model="my-model")   # raises before a mismatched query
    migrate(index, new_embed, new_model="other", switch=True)  # blue/green re-embedding

Names that need an optional extra are loaded on first use, so ``import embedstamp`` stays light::

    from embedstamp import ChromaStore        # extra "chroma"   (also: embedstamp.stores.chroma)
    from embedstamp import FastEmbedEmbedder  # extra "fastembed" (also: embedstamp.embedders.fastembed)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .embedders import FakeEmbedder, resolve_embedder
from .errors import (
    EmbeddingMismatchError,
    EmbedstampError,
    FingerprintNotFoundError,
    FingerprintTooLargeError,
    SwitchRefusedError,
)
from .errors import missing_extra_error as _missing_extra_error
from .fingerprint import Fingerprint, stamp
from .guard import Guard
from .migrate import MigrationReport, migrate, overlap_at_k
from .stores import FingerprintStore, MigratableStore, resolve_store
from .stores.local import LocalIndex
from .stores.sidecar import SidecarStore
from .verify import Thresholds, Verdict, verify

if TYPE_CHECKING:  # pragma: no cover - type checkers see the lazy names as normal imports
    from .embedders.fastembed import FastEmbedEmbedder
    from .stores.chroma import ChromaStore

__version__ = "0.1.4"

# 선택 의존성(extra)이 필요한 이름: 이름 -> (실제 모듈, 필요한 패키지, extra 이름).
# import embedstamp 할 때는 불러오지 않고, 처음 꺼내 쓰는 순간 __getattr__ 이 불러온다.
_OPTIONAL: dict[str, tuple[str, str, str]] = {
    "ChromaStore": (".stores.chroma", "chromadb", "chroma"),
    "FastEmbedEmbedder": (".embedders.fastembed", "fastembed", "fastembed"),
}


def __getattr__(name: str) -> Any:
    """Load ``ChromaStore`` / ``FastEmbedEmbedder`` on first access (PEP 562)."""
    if name not in _OPTIONAL:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, package, extra = _OPTIONAL[name]
    try:
        importlib.import_module(package)  # extra 가 없으면 여기서 실패한다
    except ImportError as exc:
        # `from embedstamp import ChromaStore` 가 "cannot import name" 대신 고치는 방법을 보여 주게 한다
        raise _missing_extra_error(name, package, extra) from exc
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value  # 다음부터는 __getattr__ 을 거치지 않는다
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_OPTIONAL])


# ChromaStore·FastEmbedEmbedder 는 extra 가 없으면 import 할 수 없으므로 `import *` 대상에서 뺀다.
__all__ = [
    "EmbeddingMismatchError",
    "EmbedstampError",
    "FakeEmbedder",
    "Fingerprint",
    "FingerprintNotFoundError",
    "FingerprintStore",
    "FingerprintTooLargeError",
    "Guard",
    "LocalIndex",
    "MigratableStore",
    "MigrationReport",
    "SidecarStore",
    "SwitchRefusedError",
    "Thresholds",
    "Verdict",
    "migrate",
    "overlap_at_k",
    "resolve_embedder",
    "resolve_store",
    "stamp",
    "verify",
]

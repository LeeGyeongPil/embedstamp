"""FastEmbedEmbedder: real local embeddings through fastembed (extra ``fastembed``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class FastEmbedEmbedder:
    """Callable wrapper around ``fastembed.TextEmbedding``.

    The first use downloads the model (set ``FASTEMBED_CACHE_PATH`` to choose where; after that
    ``HF_HUB_OFFLINE=1`` works offline). Slow first runs are normal.
    """

    def __init__(self, model: str, **kwargs: object) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "FastEmbedEmbedder needs fastembed; install it with: pip install 'embedstamp[fastembed]'"
            ) from exc
        self.model = model
        self._model = TextEmbedding(model_name=model, **kwargs)  # type: ignore[arg-type]

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)

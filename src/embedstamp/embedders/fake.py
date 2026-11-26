"""FakeEmbedder: a deterministic practice mock (no model, no network).

Each lower-cased word gets a fixed random vector; a sentence is the sum of its words, rotated
by an orthogonal matrix seeded from the *model name*, then normalised. So:

* the same model name always gives the same vectors, and sentences that share words are close
  (search really works inside one model);
* a different model name gives the same dimension but a rotated coordinate system, so vectors
  from two "models" are unrelated and nothing raises an error. This reproduces the silent
  same-dimension swap, computed fresh every time.

The numbers are not those of real models. Real measurements are in the README.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()[:8], "big")


@lru_cache(maxsize=65536)
def _token_vector(token: str, dim: int) -> np.ndarray:
    return np.random.default_rng(_seed("token", token, str(dim))).standard_normal(dim)


@lru_cache(maxsize=16)
def _rotation(model: str, dim: int) -> np.ndarray:
    gauss = np.random.default_rng(_seed("rotation", model, str(dim))).standard_normal((dim, dim))
    q, r = np.linalg.qr(gauss)
    return q * np.sign(np.diag(r))


class FakeEmbedder:
    """Callable ``list[str] -> (n, dim)`` float32 array.

    ``noise`` adds a text-seeded perturbation of relative size ``2 * noise`` (``noise=0.05``
    gives probe cosines near 0.995, the *drift* band). ``scale`` multiplies the normalised
    vectors (cosine stays the same, the norm changes). ``prefix`` is prepended to every text,
    modelling a changed prompt prefix. ``calls`` and ``texts_embedded`` count usage for tests.
    """

    def __init__(
        self,
        model: str,
        dim: int = 384,
        noise: float = 0.0,
        scale: float = 1.0,
        prefix: str = "",
    ) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        if noise < 0 or scale <= 0:
            raise ValueError("noise must be >= 0 and scale must be > 0")
        self.model, self.dim, self.noise, self.scale, self.prefix = model, dim, noise, scale, prefix
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        rows = np.stack([self._one(self.prefix + t) for t in texts]) if len(texts) else np.zeros((0, self.dim))
        return rows.astype(np.float32)

    def _one(self, text: str) -> np.ndarray:
        tokens = _WORD.findall(text.lower()) or [text]
        vec = np.sum([_token_vector(tok, self.dim) for tok in tokens], axis=0)
        vec = vec @ _rotation(self.model, self.dim)
        vec = vec / np.linalg.norm(vec)
        if self.noise:
            gen = np.random.default_rng(_seed("noise", self.model, text))
            vec = vec + 2.0 * self.noise * gen.standard_normal(self.dim) / np.sqrt(self.dim)
            vec = vec / np.linalg.norm(vec)
        return vec * self.scale

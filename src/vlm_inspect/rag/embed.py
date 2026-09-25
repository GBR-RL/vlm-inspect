"""Text embedders: a semantic model and a dependency-free lexical baseline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2; the hashing baseline uses the same width


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Returns L2-normalised row vectors, shape (len(texts), dim)."""
        ...


_WORD = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class HashingEmbedder:
    """Lexical baseline: hashed, lightly stemmed unigrams and bigrams.

    Exact-word matching only - it cannot know that "air pocket" means "bubble". It exists so the
    semantic model's gain can be measured, and so tests and CI need no model download.
    """

    name = "hashing"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [_stem(w) for w in _WORD.findall(text.lower())]
        return words + [f"{a}_{b}" for a, b in pairwise(words)]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, index] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEmbedder:
    """Semantic embeddings (sentence-transformers, all-MiniLM-L6-v2 by default; runs on CPU)."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device="cpu")
        self.name = model_id.rsplit("/", 1)[-1]
        self.dim = int(self.model.get_sentence_embedding_dimension() or EMBEDDING_DIM)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def create_embedder(kind: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("minilm", "all-MiniLM-L6-v2"):
        return SentenceTransformerEmbedder()
    return SentenceTransformerEmbedder(kind)

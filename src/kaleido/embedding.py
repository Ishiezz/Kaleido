from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np
import structlog

from kaleido.errors import EmbeddingError

log = structlog.get_logger(__name__)

_DIM = 384  # bge-small-en-v1.5 output dimension


@runtime_checkable
class TextEncoder(Protocol):
    """Interface for text embedding backends."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts to fixed-length float vectors."""
        ...

    def encode_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        ...


class BGEEmbedder:
    """Production encoder: BAAI/bge-small-en-v1.5 via sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model {model_name!r}: {exc}") from exc
        self._dim = _DIM

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vecs: np.ndarray[tuple[int, int], np.dtype[np.float32]] = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return vecs.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Encoding failed: {exc}") from exc

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class HashStubEmbedder:
    """CPU-only stub: deterministic pseudo-embeddings from MD5 hashing.

    Output is unit-normalised so cosine similarity is well-defined,
    but the distances are meaningless.  Used in CI and GPU-less demos.
    """

    @property
    def dim(self) -> int:
        return _DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vec(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        return self._hash_vec(text)

    @staticmethod
    def _hash_vec(text: str) -> list[float]:
        # Build 384 float values in [0,1] by hashing the text with different salts,
        # then L2-normalise.  Using byte values avoids NaN/Inf from raw float reinterpret.
        raw = b""
        salt = 0
        while len(raw) < _DIM:
            raw += hashlib.md5(f"{salt}:{text}".encode()).digest()
            salt += 1
        arr = np.array(list(raw[:_DIM]), dtype=np.float64) / 255.0
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        result: list[float] = arr.tolist()
        return result


def make_encoder(backend: str, model_name: str = "BAAI/bge-small-en-v1.5") -> TextEncoder:
    """Factory: returns the appropriate encoder for the configured backend."""
    if backend == "stub":
        return HashStubEmbedder()
    return BGEEmbedder(model_name)

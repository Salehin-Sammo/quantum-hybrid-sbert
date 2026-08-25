"""SBERT feature extractor with on-disk caching.

For each (task, sentence) pair the SBERT embedding (default: 384-dim) is
computed once and cached as a NumPy array keyed by SHA256(sentence). Re-runs
read from cache. This makes the train_hybrid.py pipeline fast: SBERT embedding
is a one-time cost; everything downstream (reducer + head) is what we iterate.

Default model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, 22M params,
CPU-runnable on M3 Max in seconds for 400 pairs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "sbert_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _sentence_key(sentence: str, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}|{sentence}".encode()).hexdigest()


class SBERTFeatures:
    """Cache-backed SBERT extractor. Loads model lazily.

    Usage:
        sb = SBERTFeatures(model_name="all-MiniLM-L6-v2")
        emb = sb.encode(["sentence 1", "sentence 2"])  # (N, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                  cache_dir: Path | None = None, device: str = "cpu"):
        self.model_name = model_name
        self.cache_dir = cache_dir or (CACHE_DIR / model_name.replace("/", "_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        """Encode a list of sentences with on-disk caching. Returns (N, dim) array."""
        sentences = list(sentences)
        cached = [None] * len(sentences)
        misses_idx = []
        misses_sent = []
        for i, s in enumerate(sentences):
            key = _sentence_key(s, self.model_name)
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                cached[i] = np.load(cache_file)
            else:
                misses_idx.append(i)
                misses_sent.append(s)
        if misses_sent:
            model = self._load_model()
            embeds = model.encode(misses_sent, convert_to_numpy=True,
                                    show_progress_bar=False)
            for j, idx in enumerate(misses_idx):
                key = _sentence_key(misses_sent[j], self.model_name)
                cache_file = self.cache_dir / f"{key}.npy"
                np.save(cache_file, embeds[j])
                cached[idx] = embeds[j]
        return np.array(cached, dtype=np.float32)

    def encode_pairs(self, pairs: list[tuple[str, str, float]]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode (a, b, score) tuples → (emb_a, emb_b, scores) NumPy arrays."""
        sents_a = [a for a, _, _ in pairs]
        sents_b = [b for _, b, _ in pairs]
        scores = np.array([s for _, _, s in pairs], dtype=np.float32)
        emb_a = self.encode(sents_a)
        emb_b = self.encode(sents_b)
        return emb_a, emb_b, scores


def cache_stats() -> dict:
    """Print cache statistics for debugging / progress visibility."""
    total = 0
    bytes_used = 0
    for f in CACHE_DIR.rglob("*.npy"):
        total += 1
        bytes_used += f.stat().st_size
    return {
        "cached_embeddings": total,
        "cache_size_mb": bytes_used / (1024 * 1024),
        "cache_dir": str(CACHE_DIR),
    }


if __name__ == "__main__":
    # Quick self-test
    sb = SBERTFeatures()
    emb = sb.encode(["The quick brown fox", "jumped over the lazy dog"])
    print(f"SBERT dim: {sb.dim}, emb shape: {emb.shape}")
    print(f"emb[0] norm: {np.linalg.norm(emb[0]):.4f}")
    print(f"cache: {cache_stats()}")

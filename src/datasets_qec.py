"""
Dataset loaders for the quantum-embedding-cliff study.

Each loader returns: list of (text_a, text_b, score) triples.
For STS-style: score is human similarity 0-5.
For retrieval: score is binary relevance 0/1.
"""
from __future__ import annotations

import random
from typing import List, Tuple, Iterable


Pair = Tuple[str, str, float]


def load_stsb(split: str = "validation", n_pairs: int | None = None,
              seed: int = 0) -> List[Pair]:
    from datasets import load_dataset
    ds = load_dataset("glue", "stsb", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_pairs:
        indices = indices[:n_pairs]
    return [(ds[i]["sentence1"], ds[i]["sentence2"], float(ds[i]["label"]))
            for i in indices]


def load_sick(split: str = "validation", n_pairs: int | None = None,
              seed: int = 0) -> List[Pair]:
    """SICK semantic textual similarity / entailment dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("sick", split=split, trust_remote_code=True)
    except Exception as e:
        print(f"sick load failed: {e}")
        return []
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_pairs:
        indices = indices[:n_pairs]
    return [(ds[i]["sentence_A"], ds[i]["sentence_B"], float(ds[i]["relatedness_score"]))
            for i in indices]


def load_mrpc(split: str = "validation", n_pairs: int | None = None,
              seed: int = 0) -> List[Pair]:
    """MRPC paraphrase classification — score=1 if paraphrase, 0 otherwise."""
    from datasets import load_dataset
    ds = load_dataset("glue", "mrpc", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_pairs:
        indices = indices[:n_pairs]
    return [(ds[i]["sentence1"], ds[i]["sentence2"], float(ds[i]["label"]))
            for i in indices]


def load_qqp(split: str = "validation", n_pairs: int | None = None,
             seed: int = 0) -> List[Pair]:
    """QQP duplicate question pairs."""
    from datasets import load_dataset
    ds = load_dataset("glue", "qqp", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_pairs:
        indices = indices[:n_pairs]
    return [(ds[i]["question1"], ds[i]["question2"], float(ds[i]["label"]))
            for i in indices]


def load_paws(split: str = "validation", n_pairs: int | None = None,
              seed: int = 0) -> List[Pair]:
    """PAWS — paraphrase adversarials from word scrambling."""
    try:
        from datasets import load_dataset
        ds = load_dataset("paws", "labeled_final", split=split)
    except Exception as e:
        print(f"paws load failed: {e}")
        return []
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_pairs:
        indices = indices[:n_pairs]
    return [(ds[i]["sentence1"], ds[i]["sentence2"], float(ds[i]["label"]))
            for i in indices]


REGISTRY = {
    "stsb": (load_stsb, "graded similarity 0-5"),
    "sick": (load_sick, "graded similarity 1-5"),
    "mrpc": (load_mrpc, "binary paraphrase 0/1"),
    "qqp":  (load_qqp,  "binary duplicate question 0/1"),
    "paws": (load_paws, "binary paraphrase 0/1"),
}


def load_dataset_by_name(name: str, n_pairs: int | None = None,
                         seed: int = 0, split: str = "validation") -> List[Pair]:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(REGISTRY)}")
    fn, _ = REGISTRY[name]
    return fn(split=split, n_pairs=n_pairs, seed=seed)


if __name__ == "__main__":
    for name in REGISTRY:
        try:
            pairs = load_dataset_by_name(name, n_pairs=5)
            print(f"{name:10s}  n={len(pairs):4d}  example: ({pairs[0][0][:40]!r}, {pairs[0][1][:40]!r}, {pairs[0][2]})")
        except Exception as e:
            print(f"{name:10s}  FAIL: {e}")

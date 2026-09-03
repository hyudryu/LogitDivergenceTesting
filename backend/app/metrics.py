"""Metrics for comparing next-token distributions between two serving configs.

All distribution comparisons operate on top-k logprob dictionaries keyed by
decoded token string. Tokens missing from one side's top-k set are filled with
a floor logprob and the union is renormalized. This makes truncated top-k sets
comparable at the cost of a small systematic bias, which is acceptable because
the harness is used for *relative* comparisons between cache precisions under
an identical tokenizer.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ~4e-8 probability mass assigned to tokens absent from a top-k set.
LOG_FLOOR = -17.0


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def renormalize_union(
    top_a: Dict[str, float],
    top_b: Dict[str, float],
    floor: float = LOG_FLOOR,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Project two top-k logprob dicts onto their token union and renormalize
    each into a proper probability vector."""
    tokens = sorted(set(top_a) | set(top_b))
    la = np.array([top_a.get(t, floor) for t in tokens], dtype=np.float64)
    lb = np.array([top_b.get(t, floor) for t in tokens], dtype=np.float64)
    pa = np.exp(la - logsumexp(la))
    pb = np.exp(lb - logsumexp(lb))
    return tokens, pa, pb


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in nats; 0 for identical, ln(2) for disjoint."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > eps
        return float(np.sum(a[nz] * np.log(a[nz] / b[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top1_token(top: Optional[Dict[str, float]]) -> Optional[str]:
    if not top:
        return None
    return max(top.items(), key=lambda kv: kv[1])[0]


def top1_agreement(top_a: Optional[Dict[str, float]], top_b: Optional[Dict[str, float]]) -> bool:
    return top1_token(top_a) is not None and top1_token(top_a) == top1_token(top_b)


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    nz = p > eps
    return float(-np.sum(p[nz] * np.log(p[nz])))


def compare_topk(top_a: Optional[Dict[str, float]], top_b: Optional[Dict[str, float]]) -> Optional[dict]:
    """All distribution-level metrics for a single position."""
    if not top_a or not top_b:
        return None
    _tokens, pa, pb = renormalize_union(top_a, top_b)
    return {
        "top1_agree": top1_agreement(top_a, top_b),
        "js": js_divergence(pa, pb),
        "entropy_a": entropy(pa),
        "entropy_b": entropy(pb),
        "overlap_mass": float(np.sum(np.minimum(pa, pb))),
    }


def bucket_series(
    positions: Sequence[int],
    values: Sequence[float],
    bucket_size: int,
) -> List[dict]:
    """Average a (position, value) series into fixed-size position buckets."""
    buckets: Dict[int, List[float]] = {}
    for pos, val in zip(positions, values):
        if val is None:
            continue
        key = int(pos // bucket_size) * bucket_size
        buckets.setdefault(key, []).append(float(val))
    return [
        {"bucket": b, "mean": float(np.mean(v)), "n": len(v)}
        for b, v in sorted(buckets.items())
    ]

"""Visual words: fitting vocabularies, assigning patches, counting volumes.

A patch is assigned to its nearest centroid in Euclidean distance, which for the
argmax is ``x . c - ||c||^2 / 2``. This is the rule k-means itself uses and the
one every signature is counted under; the plain inner product ``x . c`` is a
different partition and must not be used in its place.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_CHUNK = 65536


def word_scores(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``x . c_k - ||c_k||^2 / 2`` for patches ``x`` (n, D) and centroids ``C`` (K, D)."""
    C = np.asarray(C, dtype=x.dtype)
    return x @ C.T - 0.5 * (C * C).sum(1)[None, :]


def assign(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Nearest-centroid word of every patch; ``x`` may have any leading shape."""
    x = np.asarray(x)
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty(len(flat), dtype=np.int64)
    for s in range(0, len(flat), _CHUNK):
        blk = np.asarray(flat[s:s + _CHUNK], dtype=np.float32)
        out[s:s + _CHUNK] = word_scores(blk, C).argmax(1)
    return out.reshape(x.shape[:-1])


def soft_assign(x: np.ndarray, C: np.ndarray, m: int, beta: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``m`` nearest words of each patch and their weights.

    Returns ``(nearest, idx, w)``: the hard word (n,), the ``m`` nearest words
    (n, m) and weights ``w_k(p)`` proportional to ``exp(beta (x.c_k - ||c_k||^2/2))``
    normalised over those ``m`` words. ``m = 1`` is hard assignment.
    """
    s = word_scores(x, C)
    nearest = s.argmax(1)
    m = min(m, s.shape[1])
    if m == 1:
        return nearest, nearest[:, None], np.ones((len(x), 1), np.float32)
    idx = np.argpartition(-s, m - 1, axis=1)[:, :m]
    v = np.take_along_axis(s, idx, 1)
    e = np.exp((v - v.max(1, keepdims=True)) * beta)
    return nearest, idx, (e / e.sum(1, keepdims=True)).astype(np.float32)


def volumes(labels: np.ndarray, K: int) -> np.ndarray:
    """Volume vectors (N, K) from per-patch word labels (N, P): each word's share."""
    v = np.stack([np.bincount(r, minlength=K) for r in labels]).astype(np.float32)
    return v / v.sum(1, keepdims=True)


def joint_codes(labels: Sequence[np.ndarray], K: int) -> np.ndarray:
    """The joint cell ``(k_1, ..., k_V)`` of every patch, as one int64 code."""
    code = None
    for lab in labels:
        lab = np.asarray(lab, dtype=np.int64)
        code = lab.copy() if code is None else code * K + lab
    return code


def kmeans_lloyd(X: np.ndarray, K: int, seed: int = 0, iters: int = 60,
                 tol: float = 1e-5, device: str = "cpu") -> np.ndarray:
    """Lloyd iterations from ``K`` sampled points (the natural-image vocabularies)."""
    import torch
    g = torch.Generator(device=device).manual_seed(int(seed))
    Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=device)
    n = Xt.shape[0]
    C = Xt[torch.randperm(n, generator=g, device=device)[:K]].clone()
    prev = None
    for _ in range(iters):
        off = 0.5 * (C * C).sum(1)
        lab = torch.empty(n, dtype=torch.long, device=device)
        for s in range(0, n, 200_000):
            lab[s:s + 200_000] = (Xt[s:s + 200_000] @ C.T - off).argmax(1)
        newC = torch.zeros_like(C).index_add_(0, lab, Xt)
        cnt = torch.zeros(K, device=device).index_add_(
            0, lab, torch.ones(n, device=device))
        empty = cnt == 0
        newC = newC / torch.clamp(cnt, min=1.0)[:, None]
        if empty.any():                        # re-seed empty words at far points
            far = torch.topk(((Xt - newC[lab]) ** 2).sum(1), int(empty.sum()))[1]
            newC[empty] = Xt[far]
        shift = float(((newC - C) ** 2).sum(1).max())
        C = newC
        if prev is not None and shift < tol:
            break
        prev = shift
    return C.cpu().numpy().astype(np.float32)


def kmeans_plusplus(X: np.ndarray, K: int, seed: int = 0, n_init: int = 4
                    ) -> np.ndarray:
    """scikit-learn k-means++ with ``n_init`` restarts (the pathology vocabularies)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit(X)
    return km.cluster_centers_.astype(np.float32)


def fit_vocabularies(tokens: np.ndarray, K: int, n_vocabs: int = 4,
                     rule: str = "lloyd", fit_images: int = 3000,
                     sample: int = 200_000, device: str = "cpu",
                     seeds: Optional[Sequence[int]] = None) -> np.ndarray:
    """Fit ``n_vocabs`` vocabularies of ``K`` words to corpus patch tokens.

    ``tokens`` is (N, P, D). Two sampling rules are provided, one per domain:

    ``lloyd``      patches of the first ``fit_images`` images, one random sample
                   of ``sample`` patches shared by all vocabularies, Lloyd from a
                   seeded random initialisation (seeds 0..V-1).
    ``kmeans++``   for each seed, ``fit_images`` random images and ``sample`` of
                   their patches, scikit-learn k-means++ with four restarts.
    """
    seeds = list(range(n_vocabs)) if seeds is None else list(seeds)
    D = tokens.shape[-1]
    if rule == "lloyd":
        fit = np.asarray(tokens[:fit_images], dtype=np.float32).reshape(-1, D)
        sub = np.random.default_rng(0).permutation(len(fit))[:sample]
        return np.stack([kmeans_lloyd(fit[sub], K, seed=s, device=device)
                         for s in seeds])
    if rule == "kmeans++":
        out = []
        for s in seeds:
            rng = np.random.default_rng(s)
            pick = rng.permutation(len(tokens))[:fit_images]
            fit = np.asarray(tokens[pick], dtype=np.float32).reshape(-1, D)
            out.append(kmeans_plusplus(fit[rng.permutation(len(fit))[:sample]], K,
                                       seed=s))
        return np.stack(out)
    raise ValueError(f"unknown rule {rule!r}")

"""Propagation of patch scores along the encoder's own patch affinity (Psi = Phi W).

With ``u_p`` the unit-normalised token of patch ``p`` and ``s_bar`` the mean of
``u_p . u_q`` over the image,

    W_pq  proportional to  exp(gain * (u_p . u_q - threshold * s_bar)),

keeping only pairs whose argument is positive, each row normalised to sum to one,
applied once.
"""
from __future__ import annotations

import numpy as np


def affinity(tokens: np.ndarray, threshold: float = 1.2, gain: float = 3.0
             ) -> np.ndarray:
    """Row-stochastic affinity (P, P) of one image's patch tokens (P, D)."""
    f = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-12)
    sim = f @ f.T
    sim = (sim - sim.mean() * threshold) * gain
    sim[sim < 0.0] = -np.inf
    with np.errstate(invalid="ignore"):         # a row with no pair kept stays zero
        e = np.exp(sim - sim.max(1, keepdims=True))
    e[~np.isfinite(sim)] = 0.0
    return e / np.maximum(e.sum(1, keepdims=True), 1e-12)


def propagate(scores: np.ndarray, tokens: np.ndarray, threshold: float = 1.2,
              gain: float = 3.0) -> np.ndarray:
    """Apply the affinity once: ``scores`` (B, P, T), ``tokens`` (B, P, D)."""
    out = np.empty_like(scores)
    for b in range(len(scores)):
        out[b] = affinity(tokens[b], threshold, gain) @ scores[b]
    return out

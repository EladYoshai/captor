"""Counting: caption membership, signatures, permutation nulls, joint words.

Everything here reads caption strings only through set membership (which corpus
images name which term) and visual words only through their shares of an image.
No text encoder, gradient or label is involved.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import sparse

# --------------------------------------------------------------- membership

# Irregular plurals among the COCO object names; every other name takes "+s" on
# its last word.
IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives",
             "sandwich": "sandwiches", "bus": "buses", "couch": "couches",
             "skis": "skis", "scissors": "scissors", "broccoli": "broccoli",
             "sheep": "sheep", "toothbrush": "toothbrushes", "bench": "benches",
             "sink": "sinks", "glass": "glasses"}
# Compound names whose head word is itself another term.
HEAD_IS_A_TERM = {"hot dog": "dog", "teddy bear": "bear"}


def word_forms(name: str) -> tuple[str, ...]:
    """Default surface forms of a term: its name, the last word pluralised and,
    for a compound whose head names no other term, the head word alone."""
    head = name.split()[-1]
    plural = IRREGULAR.get(head, head + "s")
    other = " ".join(name.split()[:-1])
    forms = {name, (other + " " + plural).strip()}
    if other and name not in HEAD_IS_A_TERM:
        forms |= {head, plural}
    return tuple(sorted(f for f in forms if f))


def _phrase_forms(term: str, forms: Iterable[str]) -> list[str]:
    out = {f for f in forms if " " in f}
    if " " in term:
        joined = term.replace(" ", "")
        out |= {f for f in forms if f in (joined, joined + "s", joined + "es")}
        out |= {term, term + "s", term + "es"}
    return sorted(out, key=len, reverse=True)


def _contains(long: str, short: str) -> bool:
    a, b = long.split(), short.split()
    return len(b) < len(a) and any(a[i:i + len(b)] == b
                                   for i in range(len(a) - len(b) + 1))


def membership_phrase(terms: Sequence[str], texts: Sequence[str],
                      forms: dict[str, Sequence[str]],
                      strict_terms: Optional[Iterable[str]] = None,
                      min_images: int = 25) -> np.ndarray:
    """Phrase-aware caption matching, (T, N) boolean.

    ``texts[i]`` is image ``i``'s lower-cased captions joined into one string.
    Longer phrases are matched first and consumed, so the words inside a longer
    name are not available to a shorter term ("a teddy bear" does not name
    ``bear``). A bare word shared by several terms is left to the shortest of
    them. A term in ``strict_terms`` is named only by its full name, unless that
    leaves it fewer than ``min_images`` images, in which case it keeps the
    looser rule.
    """
    strict = set(strict_terms or ())
    phr = {n: _phrase_forms(n, forms[n]) for n in terms}
    owner: dict[str, list[str]] = {}
    for n in terms:
        for w in forms[n]:
            if " " not in w:
                owner.setdefault(w, []).append(n)
    words = {}
    for n in terms:
        keep = []
        for w in forms[n]:
            if " " in w or any(n2 != n and _contains(n, n2) for n2 in owner[w]):
                continue
            if n in strict and " " in n and w != n:
                continue
            keep.append(w)
        words[n] = keep
    every = sorted({(p, n) for n in terms for p in phr[n]},
                   key=lambda t: len(t[0]), reverse=True)
    M = np.zeros((len(terms), len(texts)), bool)
    idx = {n: k for k, n in enumerate(terms)}
    for c, txt in enumerate(texts):
        for p, n in every:
            if p in txt:
                M[idx[n], c] = True
                txt = txt.replace(p, " ")
        left = set(re.findall(r"[a-z]+", txt))
        for n in terms:
            if any(w in left for w in words[n]):
                M[idx[n], c] = True
    if strict:
        thin = [n for n in terms if M[idx[n]].sum() < min_images]
        if thin:
            loose = membership_phrase(terms, texts, forms, None)
            for n in thin:
                M[idx[n]] = loose[idx[n]]
    return M


def membership_substring(terms: Sequence[str], texts: Sequence[str]) -> np.ndarray:
    """A caption names a term when the term occurs in it as a substring, (T, N).

    Used for pathology terms, where "tumor" should also match "tumors" and
    "tumoral"."""
    low = [t.lower() for t in texts]
    return np.array([[t in c for c in low] for t in terms], dtype=bool)


# --------------------------------------------------------------- signatures

def corpus_moments(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corpus mean and standard deviation of each word's share."""
    return V.mean(0), V.std(0) + 1e-12


def difference_signature(V: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Eq. (signature): standardised difference of mean shares, (K, T).

    ``V`` (N, K) volume vectors, ``member`` (T, N) caption membership.
    """
    sd = V.std(0) + 1e-12
    S = np.zeros((V.shape[1], len(member)), np.float32)
    for t, m in enumerate(member):
        if m.any() and (~m).any():
            S[:, t] = (V[m].mean(0) - V[~m].mean(0)) / sd
    return S


def permutation_null(V: np.ndarray, member: np.ndarray, n_perm: int = 200,
                     seed: int = 0, pct: float = 95.0) -> np.ndarray:
    """Eq. (nullscale): the ``pct`` percentile of ``max_k |s_t[k]|`` when caption
    membership is shuffled, per term (T,). Shuffles are shared by all terms."""
    rng = np.random.default_rng(seed)
    V = np.asarray(V, np.float32)
    sd = V.std(0) + 1e-12
    M = member.astype(np.float32)
    n = len(V)
    cnt = M.sum(1, keepdims=True)
    tot = V.sum(0, keepdims=True)
    out = np.empty((len(M), n_perm), np.float32)
    for p in range(n_perm):
        inside = M[:, rng.permutation(n)] @ V
        diff = (inside / np.maximum(cnt, 1.0)
                - (tot - inside) / np.maximum(n - cnt, 1.0)) / sd
        out[:, p] = np.abs(diff).max(1)
    return np.percentile(out, pct, axis=1)


def joint_estimate(Vs: Sequence[np.ndarray], member: np.ndarray,
                   ridge: float = 0.01, n_perm: int = 200, seed: int = 0,
                   pct: float = 95.0) -> list[np.ndarray]:
    """Joint (ridge) estimation of all terms at once, each divided by its null.

    Every word's share is regressed on the indicators of all terms,
    ``B = (Yc'Yc + ridge N I)^-1 Yc' Vc / sd``, so each term keeps only the change
    it explains beyond the terms captioned with it. The null shuffles whole
    caption sets across images, keeping co-occurrence. Returns one (K, T) array
    per vocabulary.
    """
    rng = np.random.default_rng(seed)
    Y = member.T.astype(np.float64)
    Yc = Y - Y.mean(0)
    N, T = Y.shape
    reg = ridge * N * np.eye(T)
    G = np.linalg.inv(Yc.T @ Yc + reg)
    perms = [rng.permutation(N) for _ in range(n_perm)]
    Gp = [np.linalg.inv(Yc[p].T @ Yc[p] + reg) for p in perms]
    out = []
    for V in Vs:
        Vc = V - V.mean(0)
        sd = V.std(0) + 1e-12
        B = (G @ (Yc.T @ Vc)) / sd
        mx = np.stack([np.abs((g @ (Yc[p].T @ Vc)) / sd).max(1)
                       for g, p in zip(Gp, perms)], 1)
        null = np.maximum(np.percentile(mx, pct, axis=1), 1e-9)
        out.append((B / null[:, None]).T.astype(np.float32))
    return out


# --------------------------------------------------------------- joint words

class JointCells:
    """The joint partition of a corpus: cell of every patch and its statistics.

    ``codes`` (N, P) are joint codes (see :func:`captor.vocabulary.joint_codes`).
    Holds the cell-by-image volume matrix so that signatures, permutation nulls
    and caption-conditioned recounts over cells are each a sparse product.
    """

    def __init__(self, codes: np.ndarray):
        N, P = codes.shape
        self.uniq, cell = np.unique(codes.reshape(-1), return_inverse=True)
        self.cell = cell.reshape(N, P)
        self.N, self.P = N, P
        rows = np.repeat(np.arange(N), P)
        self.Vt = sparse.csr_matrix(
            (np.full(N * P, 1.0 / P, np.float32), (self.cell.reshape(-1), rows)),
            shape=(len(self.uniq), N))
        self.Vt.sum_duplicates()
        self.tot = np.asarray(self.Vt.sum(1)).ravel()
        self.n_img = np.diff(self.Vt.indptr).astype(np.float32)
        mu = self.tot / N
        dev = np.asarray(self.Vt.multiply(self.Vt).sum(1)).ravel() \
            - 2 * mu * self.tot + self.n_img * mu * mu
        self.sd = np.sqrt(np.maximum((dev + (N - self.n_img) * mu * mu) / N, 0.0)) + 1e-12

    def null(self, member: np.ndarray, n_perm: int = 200, seed: int = 0,
             pct: float = 95.0) -> np.ndarray:
        """Permutation null of ``max_c |s_t[c]|`` per term, as for words."""
        rng = np.random.default_rng(seed)
        Mt = member.T.astype(np.float32)
        cnt = Mt.sum(0)[None, :]
        out = np.empty((Mt.shape[1], n_perm), np.float32)
        for p in range(n_perm):
            ins = np.asarray(self.Vt @ Mt[rng.permutation(self.N)])
            d = (ins / np.maximum(cnt, 1.0)
                 - (self.tot[:, None] - ins) / np.maximum(self.N - cnt, 1.0))
            out[:, p] = np.abs(d / self.sd[:, None]).max(0)
        return np.maximum(np.percentile(out, pct, axis=1), 1e-9)

    def signature(self, member: np.ndarray, null: np.ndarray,
                  weights: Optional["PairWeights"] = None) -> np.ndarray:
        """Signatures over cells (C, T), divided by ``null``.

        With ``weights``, each mentioning image's patches count in proportion to
        their weight for the term (the recount of Eq. em); the contrast group is
        always counted unweighted."""
        Mt = member.T.astype(np.float32)
        cnt = Mt.sum(0)[None, :]
        out_mean = (self.tot[:, None] - np.asarray(self.Vt @ Mt)) \
            / np.maximum(self.N - cnt, 1.0)
        if weights is None:
            ins = np.asarray(self.Vt @ Mt)
        else:
            ins = np.zeros((len(self.uniq), len(member)), np.float32)
            for t in range(len(member)):
                r = weights.by_term[t]
                if len(r) == 0:
                    continue
                w = weights.W[r] / np.maximum(weights.W[r].sum(1, keepdims=True), 1e-12)
                np.add.at(ins[:, t], self.cell[weights.image[r]].reshape(-1),
                          w.reshape(-1))
        S = (ins / np.maximum(cnt, 1.0) - out_mean) / self.sd[:, None]
        return (S / null[None, :]).astype(np.float32)


# --------------------------------------------------------------- recounting

class PairWeights:
    """Per-patch weights on the (image, term) pairs that captions name."""

    def __init__(self, member: np.ndarray, P: int):
        self.term, self.image = np.nonzero(member)
        self.by_term = [np.where(self.term == t)[0] for t in range(len(member))]
        order = np.argsort(self.image, kind="stable")
        bounds = np.searchsorted(self.image[order], np.arange(member.shape[1] + 1))
        self.of_image = [order[bounds[i]:bounds[i + 1]]
                         for i in range(member.shape[1])]
        self.W = np.ones((len(self.image), P), np.float32)

    def update(self, image: int, scores: np.ndarray, tau: float, floor: float
               ) -> None:
        """Eq. (em): ``r = floor + (1 - floor) softmax_t(tau score(p, t))`` over
        the terms image ``image``'s caption names; ``scores`` (P, T)."""
        pr = self.of_image[image]
        if len(pr) == 0:
            return
        z = scores[:, self.term[pr]] * tau
        e = np.exp(z - z.max(1, keepdims=True))
        r = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
        self.W[pr] = (floor + (1.0 - floor) * r).T


def weighted_signature(labels: np.ndarray, K: int, member: np.ndarray,
                       V: np.ndarray, null: np.ndarray,
                       weights: Optional[PairWeights] = None) -> np.ndarray:
    """Per-vocabulary signatures (K, T) with mentioning images' patches weighted.

    ``labels`` (N, P) word of every corpus patch, ``V`` the unweighted volumes,
    ``null`` the per-term null of the unweighted count. With ``weights`` None this
    is Eq. (signature) divided by the null."""
    sd = V.std(0) + 1e-12
    S = np.zeros((K, len(member)), np.float32)
    P = labels.shape[1]
    for t, m in enumerate(member):
        if not m.any() or not (~m).any():
            continue
        if weights is None:
            H = V[m]
        else:
            r = weights.by_term[t]
            H = np.zeros((len(r), K), np.float32)
            np.add.at(H, (np.repeat(np.arange(len(r)), P),
                          labels[weights.image[r]].reshape(-1)),
                      weights.W[r].reshape(-1))
            H /= np.maximum(H.sum(1, keepdims=True), 1e-12)
        S[:, t] = (H.mean(0) - V[~m].mean(0)) / sd / max(float(null[t]), 1e-9)
    return S

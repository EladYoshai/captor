"""The CAPTOR read-out: term-by-patch scores ``A = S Psi`` from a counted table."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from .io import Table, load_table, resolve_table, save_table
from .propagation import propagate
from .vocabulary import soft_assign

ImageLike = Union[Image.Image, str, Path]


def patch_scores(tokens: np.ndarray, table: Table,
                 columns: Optional[np.ndarray] = None,
                 signatures: Optional[np.ndarray] = None,
                 joint_signatures: Optional[np.ndarray] = None) -> np.ndarray:
    """Score every patch for every term, (B, P, T).

    ``tokens`` (B, P, D) are unit-normalised patch tokens. Per vocabulary, each
    patch reads the signatures of its ``m`` nearest words weighted by
    ``w_k(p)`` (Eq. softscore; ``m = 1`` is the lookup of Eq. patchscore), and
    the vocabularies are averaged. With joint words, the score of the patch's
    joint cell is blended in with trust ``gamma = n / (n + lambda)`` (Eq. trust).
    Finally the map is propagated along the patch affinity (Psi = Phi W).
    ``signatures`` and ``joint_signatures`` override the table's own (used while
    a table is being counted).
    """
    ro = table.readout
    S = table.signatures if signatures is None else signatures
    SJ = table.joint_signatures if joint_signatures is None else joint_signatures
    if columns is not None:
        S = S[:, :, columns]
        SJ = None if SJ is None else SJ[:, columns]
    B, P, D = tokens.shape
    x = np.ascontiguousarray(tokens.reshape(-1, D), dtype=np.float32)
    V, K = table.n_vocabs, table.n_words
    m, beta = int(ro["soft_m"]), float(ro["soft_beta"])
    acc = np.zeros((len(x), S.shape[2]), np.float32)
    code = np.zeros(len(x), np.int64)
    for j in range(V):
        nearest, idx, w = soft_assign(x, table.centroids[j], m, beta)
        code = code * K + nearest
        for c in range(m):
            acc += w[:, c:c + 1] * S[j][idx[:, c]]
    out = acc / V
    if SJ is not None and table.has_joint and ro["trust_lambda"] is not None:
        cells, counts = table.joint_cells, table.joint_counts
        pos = np.clip(np.searchsorted(cells, code), 0, len(cells) - 1)
        hit = cells[pos] == code
        u = ((counts[pos] / (counts[pos] + float(ro["trust_lambda"]))) * hit)[:, None]
        out = (1.0 - u) * out + u * SJ[pos]
    out = out.reshape(B, P, -1)
    if ro["affinity"]:
        out = propagate(out, tokens.astype(np.float32, copy=False),
                        float(ro["affinity_threshold"]), float(ro["affinity_gain"]))
    return out


def image_scores(tokens: np.ndarray, table: Table,
                 columns: Optional[np.ndarray] = None) -> np.ndarray:
    """Eq. (score): whole-image score of each term, (B, T).

    Each vocabulary's volume vector is standardised with the caption-corpus mean
    and sd stored in the table, dotted with the signatures, and the vocabularies
    are averaged. No statistic of the images being scored enters."""
    from .vocabulary import assign, volumes
    S = table.signatures if columns is None else table.signatures[:, :, columns]
    out = 0.0
    for j in range(table.n_vocabs):
        v = volumes(assign(tokens, table.centroids[j]), table.n_words)
        z = (v - table.corpus_mean[j]) / table.corpus_sd[j]
        out = out + (z @ S[j]) / table.n_vocabs
    return out


@dataclass
class Segmentation:
    """Result of :meth:`CAPTOR.segment` on one image."""

    terms: list[str]
    labels: np.ndarray       # (G, G) index into ``terms`` of the best term per patch
    scores: np.ndarray       # (G, G, T) score of every term at every patch

    def label_names(self) -> np.ndarray:
        return np.asarray(self.terms, dtype=object)[self.labels]

    def shares(self) -> dict[str, float]:
        """Fraction of the image's patches given to each term."""
        n = self.labels.size
        return {t: float((self.labels == i).sum() / n) for i, t in enumerate(self.terms)}


class CAPTOR:
    """Text-encoder-free grounding with a counted term-by-word table.

    >>> model = CAPTOR.from_pretrained("coco_dinov2reg_k640")
    >>> seg = model.segment("photo.jpg", ["cat", "dog", "bed"])
    >>> seg.labels.shape, model.score("photo.jpg", ["cat"])
    """

    def __init__(self, table: Table, encoder=None):
        self.table = table
        self._encoder = encoder

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, Path], device: str = "cpu",
                        load_encoder: bool = True) -> "CAPTOR":
        """Load a table by pretrained name or path, and the encoder it was counted on."""
        table = load_table(resolve_table(name_or_path))
        enc = None
        if load_encoder:
            from .encoders import load_encoder as _load
            enc = _load(table.encoder, device=device)
        return cls(table, enc)

    @property
    def terms(self) -> list[str]:
        return list(self.table.terms)

    @property
    def encoder(self):
        if self._encoder is None:
            raise RuntimeError("no encoder loaded; use from_pretrained(..., "
                               "load_encoder=True) or pass tokens directly")
        return self._encoder

    def save(self, path: Union[str, Path]) -> None:
        save_table(self.table, path)

    def encode(self, images: Sequence[ImageLike]) -> np.ndarray:
        """Unit-normalised patch tokens (B, P, D) of ``images``."""
        ims = [Image.open(i).convert("RGB") if isinstance(i, (str, Path))
               else i.convert("RGB") for i in images]
        return self.encoder.tokens(ims)

    def patch_scores(self, tokens: np.ndarray,
                     terms: Optional[Sequence[str]] = None) -> np.ndarray:
        """Term-by-patch scores (B, P, T) from precomputed tokens."""
        cols = None if terms is None else self.table.term_index(list(terms))
        return patch_scores(tokens, self.table, cols)

    def segment(self, image: ImageLike, terms: Optional[Sequence[str]] = None
                ) -> Segmentation:
        """Name every patch of ``image`` with one of ``terms`` (all when None)."""
        terms = self.terms if terms is None else list(terms)
        tok = self.encode([image])
        sc = self.patch_scores(tok, terms)[0]
        G = int(round(sc.shape[0] ** 0.5))
        sc = sc.reshape(G, G, -1)
        return Segmentation(terms=terms, labels=sc.argmax(-1), scores=sc)

    def score(self, image: ImageLike, terms: Optional[Sequence[str]] = None
              ) -> dict[str, float]:
        """Whole-image score of each term (Eq. score), for ranking images."""
        terms = self.terms if terms is None else list(terms)
        s = image_scores(self.encode([image]), self.table,
                         self.table.term_index(terms))[0]
        return {t: float(v) for t, v in zip(terms, s)}

    def add_terms(self, terms: Sequence[str], corpus, forms=None) -> None:
        """Count rows for new ``terms`` from a caption corpus and append them.

        ``corpus`` is a :class:`captor.build.Corpus`, best the corpus the table
        was counted from, encoded with the table's encoder. New rows are counted
        in one pass (difference estimator, no recounting); terms too rarely
        named in the captions to bind are reported and skipped."""
        from .build import count_new_terms
        self.table = count_new_terms(self.table, corpus, list(terms), forms)

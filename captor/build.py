"""Counting a table from a captioned image corpus."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .io import Table
from .signatures import (JointCells, PairWeights, corpus_moments,
                         joint_estimate, membership_phrase, membership_substring,
                         permutation_null, weighted_signature, word_forms)
from .vocabulary import assign, fit_vocabularies, joint_codes, volumes


@dataclass
class BuildConfig:
    """Every setting of the construction (Table "hyper" of the paper)."""

    words: int = 640                 # K
    vocabs: int = 4                  # V
    kmeans: str = "lloyd"            # "lloyd" or "kmeans++" (see fit_vocabularies)
    matching: str = "phrase"         # "phrase" (natural) or "substring" (pathology)
    strict_terms: list[str] = field(default_factory=list)
    min_mentions: int = 25           # images whose caption must name a term
    min_outside: int = 25            # images whose caption must not name it
    bind_null: bool = True           # bind a term only if it beats its null
    null_scale: bool = True          # divide signatures by their null (Eq. nullscale)
    n_perm: int = 200
    estimator: str = "diff"          # "diff" (Eq. signature) or "joint" (ridge)
    ridge: float = 0.01              # joint estimate ridge, as a share of N
    joint_words: bool = True         # count over joint cells (Eq. joint)
    trust_lambda: Optional[float] = 3.0
    soft_m: int = 16
    soft_beta: float = 10.0
    em_rounds: int = 1               # caption-conditioned recounting (Eq. em)
    em_tau: float = 1.0
    em_floor: float = 0.1
    affinity: bool = True

    @classmethod
    def preset(cls, name: str) -> "BuildConfig":
        """``natural`` (COCO settings) or ``pathology`` (PathCap settings)."""
        if name == "natural":
            return cls()
        if name == "pathology":
            return cls(words=80, kmeans="kmeans++", matching="substring",
                       min_mentions=100, min_outside=1, bind_null=False, null_scale=False,
                       joint_words=False, trust_lambda=None, soft_m=1,
                       em_rounds=0)
        raise ValueError(f"unknown preset {name!r}")


@dataclass
class Corpus:
    """Patch tokens (N, P, D) of a captioned corpus and one caption string per image.

    ``tokens`` may be a memory-mapped array. For phrase matching each text should
    be the image's captions lower-cased and joined with spaces.
    """

    tokens: np.ndarray
    texts: list[str]

    @classmethod
    def encode(cls, images: Sequence, texts: Sequence[str], encoder,
               out: Optional[Path] = None, batch: int = 32,
               dtype: type = np.float16,
               progress: Optional[Callable[[int], None]] = None,
               **encode_kw) -> "Corpus":
        """Encode ``images`` (paths or PIL images) with ``encoder``.

        Tokens are kept in ``dtype`` (float16 halves the memory of a large
        corpus), in memory or in ``out`` (``.npy``), which is written under a
        temporary name and renamed only once every image is encoded."""
        import os
        from PIL import Image
        arr, tmp = None, None
        for s in range(0, len(images), batch):
            ims = [Image.open(i).convert("RGB") if isinstance(i, (str, Path))
                   else i.convert("RGB") for i in images[s:s + batch]]
            tok = encoder.tokens(ims, **encode_kw).astype(dtype)
            if arr is None:
                shape = (len(images),) + tok.shape[1:]
                if out:
                    tmp = Path(out).with_suffix(".partial.npy")
                    arr = np.lib.format.open_memmap(tmp, "w+", dtype, shape)
                else:
                    arr = np.empty(shape, dtype)
            arr[s:s + len(tok)] = tok
            if progress:
                progress(s + len(tok))
        if tmp is not None:
            arr.flush()
            del arr
            os.replace(tmp, out)
            arr = np.load(out, mmap_mode="r")
        return cls(tokens=arr, texts=list(texts))

    def labels(self, centroids: np.ndarray, chunk: int = 256) -> np.ndarray:
        """Word of every corpus patch under each vocabulary, (V, N, P)."""
        N, P = self.tokens.shape[:2]
        lab = np.empty((len(centroids), N, P), np.int32)
        for s in range(0, N, chunk):
            blk = np.asarray(self.tokens[s:s + chunk], dtype=np.float32)
            for j, C in enumerate(centroids):
                lab[j, s:s + len(blk)] = assign(blk, C)
        return lab


def _membership(cfg: BuildConfig, terms: Sequence[str], texts: Sequence[str],
                forms: dict[str, Sequence[str]]) -> np.ndarray:
    if cfg.matching == "phrase":
        return membership_phrase(terms, texts, forms, cfg.strict_terms)
    if cfg.matching == "substring":
        return membership_substring(terms, texts)
    raise ValueError(f"unknown matching rule {cfg.matching!r}")


def bind_terms(cfg: BuildConfig, terms: Sequence[str], member: np.ndarray,
               V0: np.ndarray) -> list[int]:
    """Indices of the terms that can be counted: enough images naming the term and
    not naming it and, with ``bind_null``, a strongest word above the term's own
    permutation null."""
    keep = [t for t, m in enumerate(member)
            if m.sum() >= cfg.min_mentions and (~m).sum() >= cfg.min_outside]
    if cfg.bind_null and keep:
        sub = member[keep]
        null = permutation_null(V0, sub, cfg.n_perm, seed=0)
        sd = V0.std(0) + 1e-12
        strong = [np.abs((V0[m].mean(0) - V0[~m].mean(0)) / sd).max() for m in sub]
        keep = [t for t, s, n in zip(keep, strong, null) if s > n]
    return keep


def build_table(corpus: Corpus, terms: Sequence[str], encoder: str,
                cfg: BuildConfig, forms: Optional[dict[str, Sequence[str]]] = None,
                centroids: Optional[np.ndarray] = None, device: str = "cpu",
                corpus_name: str = "", log: Callable[[str], None] = print) -> Table:
    """Fit vocabularies (unless ``centroids`` are given), count and return a table."""
    forms = {t: tuple((forms or {}).get(t, word_forms(t))) for t in terms}
    if centroids is None:
        log(f"fitting {cfg.vocabs} vocabularies of {cfg.words} words ({cfg.kmeans})")
        centroids = fit_vocabularies(corpus.tokens, cfg.words, cfg.vocabs,
                                     cfg.kmeans, device=device)
    V_, K = centroids.shape[:2]
    lab = corpus.labels(centroids)
    N, P = lab.shape[1:]
    Vs = [volumes(l, K) for l in lab]
    member_all = _membership(cfg, list(terms), corpus.texts, forms)
    keep = bind_terms(cfg, terms, member_all, Vs[0])
    dropped = [t for i, t in enumerate(terms) if i not in keep]
    if dropped:
        log(f"not bound ({len(dropped)}): {', '.join(dropped)}")
    terms = [terms[i] for i in keep]
    member = member_all[keep]
    log(f"{len(terms)} terms, {N} corpus images, {P} patches per image")

    T = len(terms)
    nulls = [permutation_null(V, member, cfg.n_perm, seed=j) if cfg.null_scale
             else np.ones(T, np.float32) for j, V in enumerate(Vs)]
    cells = JointCells(joint_codes(lab, K)) if cfg.joint_words else None
    null_c = cells.null(member, cfg.n_perm, seed=17) if cells is not None else None
    if cells is not None:
        log(f"{len(cells.uniq)} joint cells occupied")

    def count(weights: Optional[PairWeights]):
        if cfg.estimator == "joint":
            S = np.stack(joint_estimate(Vs, member, cfg.ridge, cfg.n_perm))
        else:
            S = np.stack([weighted_signature(lab[j], K, member, Vs[j], nulls[j],
                                             weights) for j in range(V_)])
        SJ = cells.signature(member, null_c, weights) if cells is not None else None
        return S, SJ

    meta = {"encoder": encoder, "corpus": corpus_name, "n_corpus": int(N),
            "config": asdict(cfg),
            "matching": {"rule": cfg.matching, "strict_terms": list(cfg.strict_terms),
                         "forms": {t: list(forms[t]) for t in terms}
                         if cfg.matching == "phrase" else {}},
            "readout": {"soft_m": cfg.soft_m, "soft_beta": cfg.soft_beta,
                        "trust_lambda": cfg.trust_lambda if cells is not None else None,
                        "affinity": cfg.affinity, "affinity_threshold": 1.2,
                        "affinity_gain": 3.0}}
    mean_, sd_ = zip(*[corpus_moments(V) for V in Vs])
    table = Table(terms=terms, centroids=centroids.astype(np.float32),
                  signatures=np.zeros((V_, K, T), np.float32),
                  corpus_mean=np.stack(mean_), corpus_sd=np.stack(sd_),
                  n_mentions=member.sum(1), meta=meta,
                  joint_cells=None if cells is None else cells.uniq,
                  joint_counts=None if cells is None else cells.n_img,
                  joint_signatures=None if cells is None
                  else np.zeros((len(cells.uniq), T), np.float32))
    table.signatures, SJ = count(None)
    if SJ is not None:
        table.joint_signatures = SJ
    rounds = cfg.em_rounds if cfg.estimator == "diff" else 0
    from .model import patch_scores
    for rd in range(rounds):
        weights = PairWeights(member, P)
        for s in range(0, N, 128):
            sc = patch_scores(np.asarray(corpus.tokens[s:s + 128], np.float32), table)
            for b in range(len(sc)):
                weights.update(s + b, sc[b], cfg.em_tau, cfg.em_floor)
        table.signatures, SJ = count(weights)
        if SJ is not None:
            table.joint_signatures = SJ
        log(f"recounting round {rd + 1} done")
    table.meta["em_rounds"] = rounds
    table.check()
    return table


def count_new_terms(table: Table, corpus: Corpus, new: list[str],
                    forms: Optional[dict[str, Sequence[str]]] = None,
                    log: Callable[[str], None] = print) -> Table:
    """A copy of ``table`` with rows for the ``new`` terms appended.

    ``corpus`` should be the corpus the table was counted from, encoded with the
    table's encoder. New rows use the table's vocabularies, matching rule, binding
    rule and null scaling, with the difference estimator and without recounting.
    Terms that do not bind are reported and left out."""
    import copy
    cfg = BuildConfig(**table.meta["config"])
    table = copy.deepcopy(table)
    new = list(dict.fromkeys(t for t in new if t not in table.terms))
    if not new:
        return table
    allf = {t: tuple(f) for t, f in table.meta["matching"].get("forms", {}).items()}
    allf.update({t: tuple((forms or {}).get(t, word_forms(t))) for t in new})
    member = _membership(cfg, list(table.terms) + new, corpus.texts,
                         allf)[len(table.terms):]
    lab = corpus.labels(table.centroids)
    K = table.n_words
    Vs = [volumes(l, K) for l in lab]
    keep = bind_terms(cfg, new, member, Vs[0])
    dropped = [t for i, t in enumerate(new) if i not in keep]
    if dropped:
        log(f"not bound ({len(dropped)}): {', '.join(dropped)}")
    if not keep:
        return table
    new, member = [new[i] for i in keep], member[keep]
    nulls = [permutation_null(V, member, cfg.n_perm, seed=j) if cfg.null_scale
             else np.ones(len(new), np.float32) for j, V in enumerate(Vs)]
    S = np.stack([weighted_signature(lab[j], K, member, Vs[j], nulls[j])
                  for j in range(table.n_vocabs)])
    table.signatures = np.concatenate([table.signatures, S], axis=2)
    if table.has_joint:
        cells = JointCells(joint_codes(lab, K))
        SJ = cells.signature(member, cells.null(member, cfg.n_perm, seed=17))
        # cells this corpus does not reach keep a zero score for the new terms
        add = np.zeros((len(table.joint_cells), len(new)), np.float32)
        pos = np.searchsorted(table.joint_cells, cells.uniq)
        ok = (pos < len(table.joint_cells)) & \
            (table.joint_cells[np.minimum(pos, len(table.joint_cells) - 1)] == cells.uniq)
        add[pos[ok]] = SJ[ok]
        table.joint_signatures = np.concatenate([table.joint_signatures, add], 1)
    table.terms = list(table.terms) + new
    table.n_mentions = np.concatenate([table.n_mentions, member.sum(1)])
    if cfg.matching == "phrase":
        table.meta["matching"]["forms"] = {t: list(allf[t]) for t in table.terms}
    table.check()
    return table

"""The CAPTOR table format and how pretrained tables are found.

A table is a single ``.npz`` file, readable with ``numpy.load`` and no pickle.

======================  ==============  =============================================
key                     shape, dtype    meaning
======================  ==============  =============================================
``meta``                (), str         JSON: encoder, read-out settings, caption
                                        matching rule and provenance (see ``Table``)
``terms``               (T,), str       term names, one column of every signature
``centroids``           (V, K, D), f4   V vocabularies of K visual words each
``signatures``          (V, K, T), f4   per-vocabulary signatures ``s_t[k]``
``corpus_mean``         (V, K), f4      caption-corpus mean of each word's share
``corpus_sd``           (V, K), f4      caption-corpus sd of each word's share
``n_mentions``          (T,), i8        corpus images whose caption names the term
``joint_cells``         (C,), i8        sorted codes of the occupied joint cells
                                        (optional; code = sum_j k_j K^(V-1-j))
``joint_counts``        (C,), f4        corpus images occupying each cell
``joint_signatures``    (C, T), f4      signatures over joint cells
======================  ==============  =============================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

FORMAT_VERSION = 1
HUB_REPO = "EladY/captor"

# name -> file name in the Hugging Face model repository ``HUB_REPO``
PRETRAINED = {
    "coco_dinov2reg_k640": "coco_dinov2reg_k640.npz",
    "pathcap_conch_k80": "pathcap_conch_k80.npz",
}

DEFAULT_READOUT = {
    "soft_m": 1,              # words a patch carries (m); 1 is hard assignment
    "soft_beta": 10.0,        # sharpness of the soft weights (beta)
    "trust_lambda": None,     # images for half belief in a joint cell (lambda)
    "affinity": True,         # propagate along the encoder's patch affinity
    "affinity_threshold": 1.2,
    "affinity_gain": 3.0,
}


@dataclass
class Table:
    """A counted term-by-word table and the settings it is read with."""

    terms: list[str]
    centroids: np.ndarray
    signatures: np.ndarray
    corpus_mean: np.ndarray
    corpus_sd: np.ndarray
    n_mentions: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    joint_cells: Optional[np.ndarray] = None
    joint_counts: Optional[np.ndarray] = None
    joint_signatures: Optional[np.ndarray] = None

    @property
    def n_vocabs(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def n_words(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def has_joint(self) -> bool:
        return self.joint_cells is not None

    @property
    def readout(self) -> dict[str, Any]:
        return {**DEFAULT_READOUT, **self.meta.get("readout", {})}

    @property
    def encoder(self) -> str:
        return str(self.meta["encoder"])

    def term_index(self, terms: Optional[list[str]] = None) -> np.ndarray:
        """Column indices of ``terms`` (all terms when None)."""
        if terms is None:
            return np.arange(len(self.terms))
        pos = {t: i for i, t in enumerate(self.terms)}
        missing = [t for t in terms if t not in pos]
        if missing:
            raise KeyError(f"terms not in the table: {missing}; count them from "
                           "captions with CAPTOR.add_terms")
        return np.array([pos[t] for t in terms], dtype=np.int64)

    def check(self) -> None:
        """Raise if the arrays are inconsistent with one another."""
        V, K, _ = self.centroids.shape
        T = len(self.terms)
        assert self.signatures.shape == (V, K, T), self.signatures.shape
        assert self.corpus_mean.shape == (V, K) == self.corpus_sd.shape
        assert self.n_mentions.shape == (T,)
        if self.has_joint:
            C = len(self.joint_cells)
            assert self.joint_counts.shape == (C,)
            assert self.joint_signatures.shape == (C, T)
            assert np.all(np.diff(self.joint_cells) > 0), "joint cells must be sorted"


def save_table(table: Table, path: str | Path) -> None:
    """Write ``table`` to ``path`` in the documented ``.npz`` format."""
    table.check()
    meta = {"format_version": FORMAT_VERSION, **table.meta}
    arrays = {
        "meta": np.array(json.dumps(meta, sort_keys=True)),
        "terms": np.array(table.terms, dtype=str),
        "centroids": table.centroids.astype(np.float32),
        "signatures": table.signatures.astype(np.float32),
        "corpus_mean": table.corpus_mean.astype(np.float32),
        "corpus_sd": table.corpus_sd.astype(np.float32),
        "n_mentions": table.n_mentions.astype(np.int64),
    }
    if table.has_joint:
        arrays["joint_cells"] = table.joint_cells.astype(np.int64)
        arrays["joint_counts"] = table.joint_counts.astype(np.float32)
        arrays["joint_signatures"] = table.joint_signatures.astype(np.float32)
    np.savez(path, **arrays)


def load_table(path: str | Path) -> Table:
    """Read a table written by :func:`save_table`."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(f"{path} uses table format {meta['format_version']}, "
                         f"newer than this package ({FORMAT_VERSION})")
    opt = {k: z[k] for k in ("joint_cells", "joint_counts", "joint_signatures")
           if k in z.files}
    table = Table(terms=[str(t) for t in z["terms"]], centroids=z["centroids"],
                  signatures=z["signatures"], corpus_mean=z["corpus_mean"],
                  corpus_sd=z["corpus_sd"], n_mentions=z["n_mentions"],
                  meta=meta, **opt)
    table.check()
    return table


def resolve_table(name_or_path: str | Path, cache_dir: Optional[str] = None) -> Path:
    """Local path of a table given a path or a pretrained name.

    A pretrained name is looked up in the directory named by the environment
    variable ``CAPTOR_TABLES``, then next to the package (``pretrained/`` in a
    source checkout), and is otherwise downloaded from the Hugging Face Hub.
    """
    p = Path(name_or_path)
    if p.suffix == ".npz" and p.exists():
        return p
    name = p.stem if p.suffix == ".npz" else str(name_or_path)
    if name not in PRETRAINED:
        raise FileNotFoundError(f"{name_or_path} is neither a file nor one of the "
                                f"pretrained tables {sorted(PRETRAINED)}")
    dirs = [Path(os.environ["CAPTOR_TABLES"])] if os.environ.get("CAPTOR_TABLES") else []
    dirs.append(Path(__file__).resolve().parents[1] / "pretrained")
    for d in dirs:
        if (d / PRETRAINED[name]).exists():
            return d / PRETRAINED[name]
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HUB_REPO, PRETRAINED[name], cache_dir=cache_dir))

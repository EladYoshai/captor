#!/usr/bin/env python3
"""Reproduce the pathology result: BCSS, 62 held-out patients, five tissues.

    python scripts/eval_bcss.py --bcss-root /data/bcss --table pathcap_conch_k80

``--bcss-root`` holds ``rgbs_colorNormalized/``, ``masks/`` and
``meta/gtruth_codes.tsv`` as distributed by the BCSS authors (0.25 MPP).
With the CONCH table this prints mIoU 0.4966 on the reporting patients.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from captor import CAPTOR
from captor.evaluation import (BCSS_TERMS, bcss_cell_truth, bcss_codes,
                               bcss_fields, miou, split_half)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bcss-root", type=Path, required=True)
    ap.add_argument("--table", default="pathcap_conch_k80")
    ap.add_argument("--split", default="test", choices=("dev", "test", "all"))
    ap.add_argument("--tile", type=int, default=896)
    ap.add_argument("--max-fields", type=int, default=200)
    ap.add_argument("--dev-frac", type=float, default=0.4)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--tokens", type=Path, default=None,
                    help="cache of the fields' patch tokens (.npy); written if absent")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    fields = bcss_fields(a.bcss_root, a.tile, a.max_fields)
    codes = bcss_codes(a.bcss_root)
    if a.tokens and a.tokens.exists():
        model = CAPTOR.from_pretrained(a.table, load_encoder=False)
        tok = np.load(a.tokens)
    else:
        model = CAPTOR.from_pretrained(a.table, device=a.device)
        crops = [Image.open(rp).convert("RGB").crop((x, y, x + a.tile, y + a.tile))
                 for rp, _, x, y in fields]
        tok = model.encoder.tokens(crops)
        if a.tokens:
            np.save(a.tokens, tok)
    G = int(round(tok.shape[1] ** 0.5))
    truth = np.stack([bcss_cell_truth(_mask(a.bcss_root, rp, x, y, a.tile), G, codes)
                      for rp, _, x, y in fields])

    pats = np.array([p for _, p, _, _ in fields])
    up = np.unique(pats)
    dev_pat = set(up[split_half(len(up), a.dev_frac, a.split_seed)].tolist())
    sel = np.array([a.split == "all" or (p in dev_pat) == (a.split == "dev")
                    for p in pats])
    pred = model.patch_scores(tok[sel], BCSS_TERMS).argmax(-1).ravel()
    t = truth[sel].ravel()
    ok = t >= 0
    print(f"{a.split}: {sel.sum()} fields from {len(set(pats[sel]))} patients, "
          f"{ok.sum()} labelled cells")
    print(f"mIoU {miou(pred[ok], t[ok], len(BCSS_TERMS)):.4f}  "
          f"accuracy {(pred[ok] == t[ok]).mean():.4f}")


def _mask(root: Path, rp: Path, x: int, y: int, tile: int) -> np.ndarray:
    m = np.asarray(Image.open(root / "masks" / rp.name))
    m = m[..., 0] if m.ndim == 3 else m
    return m[y:y + tile, x:x + tile]


if __name__ == "__main__":
    main()

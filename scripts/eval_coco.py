#!/usr/bin/env python3
"""Reproduce the natural-image result: COCO 78-way grid mIoU, reporting half.

    python scripts/eval_coco.py --coco-root /data/coco --table coco_dinov2reg_k640

``--coco-root`` holds ``val2017/`` and ``annotations/instances_val2017.json``.
The first 5,000 val2017 images are read at 196 pixels; scored on the 2,500 of
the reporting half, this prints mIoU 0.5654.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from captor import CAPTOR
from captor.evaluation import coco_eval_images, coco_grid_truth, miou, split_half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-root", type=Path, required=True)
    ap.add_argument("--table", default="coco_dinov2reg_k640")
    ap.add_argument("--eval-images", type=int, default=5000)
    ap.add_argument("--dev-frac", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=("dev", "test", "all"))
    ap.add_argument("--tokens", type=Path, default=None,
                    help="cache of the evaluation images' patch tokens (.npy); "
                         "written if absent")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from pycocotools.coco import COCO
    ins = COCO(str(a.coco_root / "annotations/instances_val2017.json"))
    ids = coco_eval_images(ins, a.eval_images)
    have_tok = a.tokens is not None and a.tokens.exists()
    model = CAPTOR.from_pretrained(a.table, device=a.device, load_encoder=not have_tok)
    if have_tok:
        tok = np.load(a.tokens, mmap_mode="r")
    else:
        paths = [a.coco_root / "val2017" / ins.loadImgs(i)[0]["file_name"] for i in ids]
        tok = np.concatenate([model.encode(paths[s:s + a.batch]).astype(np.float16)
                              for s in range(0, len(paths), a.batch)])
        if a.tokens:
            np.save(a.tokens, tok)
    G = int(round(tok.shape[1] ** 0.5))

    # only the table's terms that are COCO categories compete for a cell
    cat = {c["name"]: c["id"] for c in ins.loadCats(ins.getCatIds())}
    terms = [t for t in model.terms if t in cat]
    classes = [(t, cat[t]) for t in terms]
    truth = np.stack([coco_grid_truth(ins, i, G, classes).ravel() for i in ids])

    pred = np.concatenate([
        model.patch_scores(np.asarray(tok[s:s + a.batch], np.float32), terms).argmax(-1)
        for s in range(0, len(ids), a.batch)])
    dev = split_half(len(ids), a.dev_frac, a.split_seed)
    sel = np.ones(len(ids), bool) if a.split == "all" else (dev if a.split == "dev" else ~dev)
    p, t = pred[sel].ravel(), truth[sel].ravel()
    ok = t >= 0
    print(f"{a.split}: {sel.sum()} images, {ok.sum()} labelled cells")
    print(f"mIoU {miou(p[ok], t[ok], len(terms)):.4f}  "
          f"accuracy {(p[ok] == t[ok]).mean():.4f}")


if __name__ == "__main__":
    main()

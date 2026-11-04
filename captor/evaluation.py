"""The two evaluation protocols of the paper: COCO grid mIoU and BCSS fields.

COCO: a 14 x 14 grid over the centre crop of the first 5,000 val2017 images; a
cell is labelled only when one class covers more than half of it, and there is
no background class. The 5,000 images are split in half by image (seed 0); the
reported number is the mIoU on the reporting half.

BCSS: 200 fields of 896 x 896 pixels from the colour-normalised regions, each
read at 224 pixels; a cell takes the majority label among the five tissues;
patients are split 41 for development and 62 for reporting (seed 0).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def miou(pred: np.ndarray, truth: np.ndarray, n: int) -> float:
    """Mean IoU over the classes present in ``truth``."""
    out = []
    for k in range(n):
        p, t = pred == k, truth == k
        if t.sum() == 0:
            continue
        u = (p | t).sum()
        out.append(float((p & t).sum() / u) if u else 0.0)
    return float(np.mean(out))


def split_half(n: int, frac: float, seed: int = 0) -> np.ndarray:
    """Boolean development mask over ``n`` units: a seeded permutation, first
    ``round(frac n)`` units are development."""
    perm = np.random.default_rng(seed).permutation(n)
    dev = np.zeros(n, bool)
    dev[perm[:int(round(frac * n))]] = True
    return dev


# ------------------------------------------------------------------- COCO

def coco_eval_images(instances, n: int = 5000) -> list[int]:
    """The evaluation images: the first ``n`` val2017 ids in sorted order."""
    return sorted(instances.getImgIds())[:n]


def coco_grid_truth(instances, img_id: int, G: int,
                    classes: Sequence[tuple[str, int]]) -> np.ndarray:
    """Class index of each cell of a G x G grid over the centre crop, -1 if none."""
    info = instances.loadImgs(img_id)[0]
    side = min(info["height"], info["width"])
    top, left = (info["height"] - side) // 2, (info["width"] - side) // 2
    truth = np.full((G, G), -1, np.int16)
    best = np.zeros((G, G))
    e = np.linspace(0, side, G + 1).astype(int)
    for k, (_, cid) in enumerate(classes):
        anns = instances.loadAnns(instances.getAnnIds(imgIds=img_id, catIds=[cid],
                                                      iscrowd=None))
        if not anns:
            continue
        mm = np.zeros((info["height"], info["width"]), bool)
        for an in anns:
            mm |= instances.annToMask(an).astype(bool)
        mc = mm[top:top + side, left:left + side]
        if not mc.any():
            continue
        for r in range(G):
            for c in range(G):
                f = mc[e[r]:e[r + 1], e[c]:e[c + 1]].mean()
                if f > best[r, c] and f > 0.5:
                    best[r, c], truth[r, c] = f, k
    return truth


# ------------------------------------------------------------------- BCSS

BCSS_TERMS = ["tumor", "stroma", "lymphocyte", "necrosis", "adipose"]
BCSS_CLASS = {"tumor": "tumor", "stroma": "stroma",
              "lymphocyte": "lymphocytic_infiltrate",
              "necrosis": "necrosis_or_debris", "adipose": "fat"}
BCSS_DROP = {0, 7, 15}          # outside the ROI, glandular secretions, blood vessel


def bcss_fields(root: Path, tile: int = 896, max_fields: int = 200, seed: int = 0):
    """Sample the evaluation fields: (rgb path, patient, x, y) for each.

    Every non-overlapping ``tile`` window whose pixels are at most 20% in the
    dropped classes is a candidate; candidates are shuffled with ``seed`` and the
    first ``max_fields`` kept.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    pool = []
    for rp in sorted((root / "rgbs_colorNormalized").glob("*.png")):
        mp = root / "masks" / rp.name
        if not mp.exists():
            continue
        pid = "-".join(rp.name.split("-")[:3])
        msk = np.asarray(Image.open(mp))
        msk = msk[..., 0] if msk.ndim == 3 else msk
        H, W = msk.shape
        for y in range(0, H - tile + 1, tile):
            for x in range(0, W - tile + 1, tile):
                if np.isin(msk[y:y + tile, x:x + tile], list(BCSS_DROP)).mean() > 0.2:
                    continue
                pool.append((rp, pid, x, y))
    np.random.default_rng(seed).shuffle(pool)
    return pool[:max_fields]


def bcss_codes(root: Path) -> dict[int, str]:
    codes = {}
    for line in (root / "meta/gtruth_codes.tsv").read_text().splitlines()[1:]:
        if line.strip():
            lab, c = line.split("\t")
            codes[int(c)] = lab
    return codes


def bcss_cell_truth(mask: np.ndarray, G: int, codes: dict[int, str],
                    terms: Sequence[str] = BCSS_TERMS) -> np.ndarray:
    """Majority term index of each grid cell of a field mask, -1 if none."""
    want = {BCSS_CLASS[t]: i for i, t in enumerate(terms)}
    lut = np.full(256, -1, np.int64)
    for v, name in codes.items():
        if name in want:
            lut[v] = want[name]
    step = mask.shape[0] // G
    out = np.full(G * G, -1, np.int64)
    for r in range(G):
        for c in range(G):
            lab = lut[mask[r * step:(r + 1) * step, c * step:(c + 1) * step].ravel()]
            lab = lab[lab >= 0]
            if len(lab):
                vals, cnts = np.unique(lab, return_counts=True)
                out[r * G + c] = vals[cnts.argmax()]
    return out

"""Parity with the paper's two headline numbers, from cached features.

These tests need the datasets and are skipped unless the paths are given:

CAPTOR_COCO_EVAL_DIR   directory holding ``eval_tokens.npy`` (5000, 196, 768),
                       the DINOv2-reg tokens of the first 5,000 val2017 images at
                       196 px, and ``truth.npy``, their grid truth (written by
                       ``scripts/eval_coco.py --tokens``, or built here from
                       CAPTOR_COCO_ROOT)
CAPTOR_COCO_ROOT       optional COCO root, to rebuild and check the grid truth
CAPTOR_COCO_REFERENCE  optional reference predictions over the labelled cells
CAPTOR_COCO_TABLE      optional path of the COCO table (default: pretrained)
CAPTOR_BCSS_ROOT       BCSS root (``rgbs_colorNormalized``, ``masks``, ``meta``)
CAPTOR_BCSS_TOKENS     cached CONCH tokens of the 200 fields (``eval_bcss.py --tokens``)
"""
import os
from pathlib import Path

import numpy as np
import pytest

from captor import CAPTOR
from captor.evaluation import (BCSS_TERMS, bcss_cell_truth, bcss_codes, bcss_fields,
                               coco_eval_images, coco_grid_truth, miou, split_half)


def _env(name):
    v = os.environ.get(name)
    return Path(v) if v else None


COCO = _env("CAPTOR_COCO_EVAL_DIR")
BCSS = _env("CAPTOR_BCSS_ROOT")
BCSS_TOK = _env("CAPTOR_BCSS_TOKENS")


@pytest.fixture(scope="module")
def coco_preds():
    model = CAPTOR.from_pretrained(os.environ.get("CAPTOR_COCO_TABLE",
                                                  "coco_dinov2reg_k640"),
                                   load_encoder=False)
    tok = np.load(COCO / "eval_tokens.npy", mmap_mode="r")
    pred = np.concatenate([
        model.patch_scores(np.asarray(tok[s:s + 100], np.float32)).argmax(-1)
        for s in range(0, len(tok), 100)])
    return model, pred.ravel()


@pytest.mark.skipif(COCO is None, reason="CAPTOR_COCO_EVAL_DIR not set")
def test_coco_headline(coco_preds):
    model, pred = coco_preds
    truth = np.load(COCO / "truth.npy").astype(np.int64)
    n = len(truth) // 196
    test = ~np.repeat(split_half(n, 0.5, 0), 196)
    ok = truth >= 0
    m = miou(pred[ok & test], truth[ok & test], len(model.terms))
    acc = (pred[ok & test] == truth[ok & test]).mean()
    assert round(m, 4) == 0.5654, m
    assert round(acc, 3) == 0.768, acc
    ref = _env("CAPTOR_COCO_REFERENCE")
    if ref is not None:
        agree = (np.load(ref) == pred[ok]).mean()
        assert agree > 0.9995, agree


@pytest.mark.skipif(COCO is None or _env("CAPTOR_COCO_ROOT") is None,
                    reason="CAPTOR_COCO_EVAL_DIR or CAPTOR_COCO_ROOT not set")
def test_coco_grid_truth(coco_preds):
    from pycocotools.coco import COCO as C
    model, _ = coco_preds
    ins = C(str(_env("CAPTOR_COCO_ROOT") / "annotations/instances_val2017.json"))
    cat = {c["name"]: c["id"] for c in ins.loadCats(ins.getCatIds())}
    classes = [(t, cat[t]) for t in model.terms]
    ids = coco_eval_images(ins)[:100]
    mine = np.concatenate([coco_grid_truth(ins, i, 14, classes).ravel() for i in ids])
    np.testing.assert_array_equal(mine, np.load(COCO / "truth.npy")[:len(mine)])


@pytest.mark.skipif(BCSS is None or BCSS_TOK is None,
                    reason="CAPTOR_BCSS_ROOT or CAPTOR_BCSS_TOKENS not set")
@pytest.mark.parametrize("split,expected", [("test", 0.4966), ("dev", 0.4521)])
def test_bcss_headline(split, expected):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    model = CAPTOR.from_pretrained("pathcap_conch_k80", load_encoder=False)
    tok = np.load(BCSS_TOK)
    fields = bcss_fields(BCSS)
    codes = bcss_codes(BCSS)
    G = int(round(tok.shape[1] ** 0.5))
    truth = []
    for rp, _, x, y in fields:
        m = np.asarray(Image.open(BCSS / "masks" / rp.name))
        m = m[..., 0] if m.ndim == 3 else m
        truth.append(bcss_cell_truth(m[y:y + 896, x:x + 896], G, codes))
    truth = np.stack(truth)
    pats = np.array([p for _, p, _, _ in fields])
    up = np.unique(pats)
    assert len(up) == 103
    dev = set(up[split_half(len(up), 0.4, 0)].tolist())
    sel = np.array([(p in dev) == (split == "dev") for p in pats])
    pred = model.patch_scores(tok[sel], BCSS_TERMS).argmax(-1).ravel()
    t = truth[sel].ravel()
    ok = t >= 0
    assert round(miou(pred[ok], t[ok], 5), 4) == expected

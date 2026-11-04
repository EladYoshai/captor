#!/usr/bin/env python3
"""Name every patch of an image with one of a list of terms and draw the result.

    python scripts/segment.py --table coco_dinov2reg_k640 --image photo.jpg \\
        --terms cat,dog,bed --out photo_captor.png

Saves the image, its colour-coded label map with a legend and, with
``--maps``, one score map per term.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from captor import CAPTOR
from captor.viz import crop_like_encoder, heatmap, legend, overlay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="coco_dinov2reg_k640")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--terms", default="", help="comma-separated; default all terms")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--maps", action="store_true", help="also draw each term's scores")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    model = CAPTOR.from_pretrained(a.table, device=a.device)
    terms = [t.strip() for t in a.terms.split(",") if t.strip()] or None
    image = Image.open(a.image).convert("RGB")
    seg = model.segment(image, terms)
    view = crop_like_encoder(image, model.table.encoder == "dinov2_reg")
    view.thumbnail((768, 768))

    for t, s in sorted(seg.shares().items(), key=lambda kv: -kv[1]):
        if s > 0:
            print(f"{t:<20}{100 * s:5.1f}% of patches")
    panels = [view, overlay(view, seg.labels)]
    titles = ["image", "CAPTOR"]
    if a.maps:
        for i, t in enumerate(seg.terms):
            panels.append(Image.blend(view, heatmap(seg.scores[..., i], view.size), 0.6))
            titles.append(t)
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.6))
    for x, im, ti in zip(ax, panels, titles):
        x.imshow(im)
        x.set_title(ti)
        x.axis("off")
    used = set(seg.labels.ravel().tolist())
    handles = [Patch(color=[c / 255 for c in col], label=t)
               for i, (t, col) in enumerate(legend(seg.terms)) if i in used]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = a.out or a.image.with_name(a.image.stem + "_captor.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""CAPTOR on natural images: DINOv2 with registers and the COCO caption table.

    python examples/natural_images.py

Downloads the table (about 67 MB) and DINOv2 on first use, runs on the CPU.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from captor import CAPTOR
from captor.viz import crop_like_encoder, heatmap, legend, overlay

HERE = Path(__file__).parent
# Each photograph with the COCO objects that appear in it. Every patch is named
# with one of these terms, so list the objects you expect to find.
EXAMPLES = {"giraffe_zebra.jpg": ["giraffe", "zebra"],
            "beagle_sofa.jpg": ["dog"],
            "elephants.jpg": ["elephant"],
            "cyclist.jpg": ["bicycle", "person", "car"],
            "street.jpg": ["bicycle", "person", "car"],
            "breakfast.jpg": ["dining table", "sandwich", "cup", "bottle"]}

model = CAPTOR.from_pretrained("coco_dinov2reg_k640")
print(f"{len(model.terms)} terms, {model.table.n_vocabs} vocabularies of "
      f"{model.table.n_words} words")

fig, ax = plt.subplots(3, len(EXAMPLES), figsize=(3.6 * len(EXAMPLES), 12.5))
for c, (name, terms) in enumerate(EXAMPLES.items()):
    image = Image.open(HERE / "images" / name).convert("RGB")
    seg = model.segment(image, terms)                 # one term per patch
    view = crop_like_encoder(image, center_crop=True)  # DINOv2 reads the centre square
    keep = seg.confident(0.3)        # patches where a listed term scores well
    shares = {t: round(float((seg.labels[keep] == i).mean()), 2) for i, t in enumerate(terms)}
    print(f"{name:<18}", shares)
    ax[0, c].imshow(view)
    ax[0, c].set_title(name, fontsize=10)
    # faded patches: no listed term scores well there (sky, road, grass)
    ax[1, c].imshow(overlay(view, seg.labels, keep=keep))
    ax[1, c].legend(handles=[Patch(color=[v / 255 for v in col], label=t)
                             for t, col in legend(terms)],
                    loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
                    fontsize=9, frameon=False)
    ax[2, c].imshow(heatmap(seg.scores[..., 0], view.size))   # the first term's row of A
    ax[2, c].set_title(f"score map: {terms[0]}", fontsize=10)
for a in ax.ravel():
    a.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.06, hspace=0.32)
fig.savefig(HERE / "natural_images_output.png", dpi=80)
print("wrote examples/natural_images_output.png")

# Whole-image scores rank images by a term
for term in ["giraffe", "dog", "bicycle", "cup"]:
    s = {n: model.score(HERE / "images" / n, [term])[term] for n in EXAMPLES}
    print(f"{term:<8} best match: {max(s, key=s.get)}")

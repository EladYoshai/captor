"""CAPTOR on natural images: DINOv2 with registers and the COCO caption table.

    python examples/natural_images.py

Downloads the table (about 67 MB) and DINOv2 on first use, runs on the CPU.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from captor import CAPTOR
from captor.viz import crop_like_encoder, heatmap, overlay

HERE = Path(__file__).parent
EXAMPLES = {"cat_lizard.jpg": ["cat", "dog", "bird", "bench", "potted plant"],
            "girl_dog.jpg": ["person", "dog", "cat", "car", "bench"],
            "pizza.jpg": ["pizza", "dining table", "cup", "fork", "person"],
            "horse.jpg": ["horse", "cow", "sheep", "person", "car"]}

model = CAPTOR.from_pretrained("coco_dinov2reg_k640")
print(f"{len(model.terms)} terms, {model.table.n_vocabs} vocabularies of "
      f"{model.table.n_words} words")

fig, ax = plt.subplots(3, len(EXAMPLES), figsize=(4 * len(EXAMPLES), 12))
for c, (name, terms) in enumerate(EXAMPLES.items()):
    image = Image.open(HERE / "images" / name).convert("RGB")
    seg = model.segment(image, terms)                 # one term per patch
    rank = model.score(image, terms)                  # whole-image scores
    view = crop_like_encoder(image, center_crop=True)
    print(name, {t: round(s, 2) for t, s in seg.shares().items() if s > 0},
          "top term:", max(rank, key=rank.get))
    ax[0, c].imshow(view)
    ax[1, c].imshow(overlay(view, seg.labels))
    ax[1, c].set_title(", ".join(t for t in terms if seg.shares()[t] > 0.05))
    ax[2, c].imshow(heatmap(seg.scores[..., 0], view.size))   # the first term's row of A
    ax[2, c].set_title(f"score map: {terms[0]}")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig(HERE / "natural_images_output.png", dpi=90)
print("wrote examples/natural_images_output.png")

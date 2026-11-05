"""CAPTOR on histology: the CONCH image encoder and the PathCap caption table.

    python examples/pathology.py

CONCH is gated: request access at https://huggingface.co/MahmoodLab/CONCH, run
``huggingface-cli login`` and ``pip install git+https://github.com/Mahmoodlab/CONCH.git``.
For UNI or Phikon-v2, encoders without a text tower, build a table first with
``scripts/build_table.py --preset pathology --encoder uni`` (or ``phikon_v2``).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from captor import CAPTOR
from captor.viz import heatmap, legend, overlay

HERE = Path(__file__).parent
TERMS = ["tumor", "stroma", "lymphocyte", "necrosis", "adipose"]
FIELDS = sorted((HERE / "images").glob("bcss_*.jpg"))

model = CAPTOR.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "pathcap_conch_k80")
fig, ax = plt.subplots(3, len(FIELDS), figsize=(4 * len(FIELDS), 12.5))
for c, path in enumerate(FIELDS):
    field = Image.open(path).convert("RGB")
    seg = model.segment(field, TERMS)
    print(path.name, {t: round(s, 2) for t, s in seg.shares().items()})
    ax[0, c].imshow(field)
    ax[1, c].imshow(overlay(field, seg.labels))
    ax[2, c].imshow(heatmap(seg.scores[..., TERMS.index("tumor")], field.size))
    ax[2, c].set_title("score map: tumor")
for a in ax.ravel():
    a.axis("off")
fig.legend(handles=[Patch(color=[v / 255 for v in col], label=t)
                    for t, col in legend(TERMS)], loc="lower center", ncol=5,
           frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(HERE / "pathology_output.png", dpi=90)
print("wrote examples/pathology_output.png")

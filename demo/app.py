"""CAPTOR demo: name every patch of an image with terms counted from captions.

    python demo/app.py

Natural images use DINOv2 with registers and the COCO caption table. Pathology
uses CONCH (or UNI / Phikon-v2) with the PathCap caption table; the licences of
these encoders do not cover a public hosted demo, so pathology runs only when the
app is started on your own machine with your own Hugging Face access.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image

from captor import CAPTOR
from captor.viz import crop_like_encoder, heatmap, legend, overlay

HERE = Path(__file__).resolve().parent
EXAMPLES = next((p for p in (HERE / "examples", HERE.parent / "examples" / "images")
                 if p.exists()), HERE)
HOSTED = bool(os.environ.get("SPACE_ID"))
NATURAL, PATHOLOGY = "Natural images", "Pathology"
PATH_TABLES = {"CONCH": "pathcap_conch_k80"}
DEFAULT_TERMS = {NATURAL: ["giraffe", "zebra"],
                 PATHOLOGY: ["tumor", "stroma", "lymphocyte", "necrosis", "adipose"]}
# the objects in each bundled photograph, filled in when the example is picked
EXAMPLE_TERMS = {"giraffe_zebra.jpg": ["giraffe", "zebra"],
                 "beagle_sofa.jpg": ["dog"],
                 "elephants.jpg": ["elephant"],
                 "cyclist.jpg": ["bicycle", "person", "car"],
                 "street.jpg": ["bicycle", "person", "car"],
                 "breakfast.jpg": ["dining table", "sandwich", "cup", "bottle"]}
DEFAULT_FADE = {NATURAL: 0.3, PATHOLOGY: 0.0}
PATHOLOGY_NOTE = (
    "Pathology mode is not available in the hosted demo. CONCH and UNI are gated "
    "and licensed for non-commercial research use by registered users, and the "
    "Phikon-v2 licence is non-commercial; none of them clearly permits serving "
    "the model to the public. Run this app on your own machine after requesting "
    "access on the model pages and logging in with `huggingface-cli login`:\n\n"
    "`git clone https://github.com/EladYoshai/captor.git && cd captor && "
    "pip install -e \".[pathology]\" gradio && python demo/app.py`")


@lru_cache(maxsize=4)
def get_model(table: str) -> CAPTOR:
    return CAPTOR.from_pretrained(table, device="cuda" if _cuda() else "cpu")


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def table_for(domain: str, encoder: str) -> str:
    return "coco_dinov2reg_k640" if domain == NATURAL else PATH_TABLES[encoder]


def term_choices(domain: str, encoder: str) -> list[str]:
    from captor.io import load_table, resolve_table
    if domain == PATHOLOGY and HOSTED:
        return DEFAULT_TERMS[PATHOLOGY]
    return load_table(resolve_table(table_for(domain, encoder))).terms


def example_files() -> list[Path]:
    """Bundled images, the photographs in the order of EXAMPLE_TERMS."""
    order = list(EXAMPLE_TERMS)
    return sorted(EXAMPLES.glob("*.jpg"),
                  key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name))


def on_domain(domain: str, encoder: str):
    path = domain == PATHOLOGY
    ex = [p for p in example_files() if p.name.startswith("bcss_") == path]
    return (gr.update(choices=term_choices(domain, encoder),
                      value=DEFAULT_TERMS[domain]),
            gr.update(visible=path and not HOSTED),
            gr.update(value=PATHOLOGY_NOTE if path and HOSTED else "",
                      visible=path and HOSTED),
            gr.update(value=[str(p) for p in ex]),
            None,
            gr.update(value=DEFAULT_FADE[domain]))


def run(domain: str, encoder: str, image, terms: list[str], fade: float = 0.0):
    if image is None:
        raise gr.Error("Upload an image or pick an example first.")
    if domain == PATHOLOGY and HOSTED:
        raise gr.Error("Pathology runs only locally; see the note above.")
    if not terms:
        raise gr.Error("Pick at least one term.")
    try:
        model = get_model(table_for(domain, encoder))
    except Exception as e:                      # gated weights without access
        raise gr.Error(f"Could not load the encoder ({type(e).__name__}: {e}). "
                       "For CONCH and UNI, request access on Hugging Face and log in.")
    image = image.convert("RGB")
    seg = model.segment(image, terms)
    view = crop_like_encoder(image, domain == NATURAL)
    view.thumbnail((640, 640))
    keep = seg.confident(fade) if fade > 0 else None
    shares = seg.shares() if keep is None else {
        t: float((seg.labels[keep] == i).mean()) for i, t in enumerate(seg.terms)}
    chips = "".join(
        f'<span style="display:inline-block;margin:2px 8px 2px 0">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:rgb{col};border-radius:2px;margin-right:4px"></span>'
        f'{t} {100 * shares[t]:.0f}%</span>'
        for t, col in legend(seg.terms) if shares[t] > 0)
    maps = [(Image.blend(view, heatmap(seg.scores[..., i], view.size), 0.6), t)
            for i, t in enumerate(seg.terms)]
    return overlay(view, seg.labels, keep=keep), chips, maps


def pick_example(evt: gr.SelectData, files, terms):
    f = files[evt.index]
    path = Path(f[0] if isinstance(f, (list, tuple)) else f)
    for name, t in EXAMPLE_TERMS.items():
        if path.name.endswith(name):
            terms = t
    return Image.open(path), terms


with gr.Blocks(title="CAPTOR") as demo:
    gr.Markdown(
        "# CAPTOR\n"
        "Text-encoder-free grounding: each term is a row of a table counted from "
        "captions over a frozen image encoder's visual words, and every patch is "
        "named by looking that table up. No text encoder is used.")
    with gr.Row():
        domain = gr.Radio([NATURAL, PATHOLOGY], value=NATURAL, label="1. Domain")
        encoder = gr.Radio(list(PATH_TABLES), value="CONCH", label="Pathology encoder",
                           visible=False)
    note = gr.Markdown(visible=False)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="2. Image")
            gallery = gr.Gallery(value=[str(p) for p in example_files()
                                        if not p.name.startswith("bcss_")],
                                 label="or pick an example", columns=4, height=140,
                                 allow_preview=False)
            terms = gr.Dropdown(choices=term_choices(NATURAL, "CONCH"),
                                value=DEFAULT_TERMS[NATURAL], multiselect=True,
                                label="3. Terms (the objects you expect in the image)")
            fade = gr.Slider(0.0, 0.6, value=DEFAULT_FADE[NATURAL], step=0.05,
                             label="Fade patches where no term scores well "
                                   "(fraction of the top score, display only)")
            go = gr.Button("Segment", variant="primary")
        with gr.Column():
            labels = gr.Image(label="Label map", type="pil")
            shares = gr.HTML()
            maps = gr.Gallery(label="Score map of each term (red high, blue low)",
                              columns=3, height=300)
    domain.change(on_domain, [domain, encoder],
                  [terms, encoder, note, gallery, image, fade])
    encoder.change(lambda d, e: gr.update(choices=term_choices(d, e)),
                   [domain, encoder], terms)
    gallery.select(pick_example, [gallery, terms], [image, terms])
    go.click(run, [domain, encoder, image, terms, fade], [labels, shares, maps])
    gr.Markdown(
        "Natural images: DINOv2 ViT-B/14 with registers at 196 pixels (centre crop), "
        "table counted from 15,000 COCO captions, 78 object terms. Pathology: a "
        "field read at 224 pixels, table counted from 32,463 PathCap captions. "
        "Example photographs are from Wikimedia Commons (CC BY-SA or CC0, authors in "
        "the repository's image credits) and histology fields from BCSS (CC BY 4.0).")

if __name__ == "__main__":
    demo.launch()

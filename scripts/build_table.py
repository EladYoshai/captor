#!/usr/bin/env python3
"""Count a CAPTOR table from a folder of images and their captions.

    python scripts/build_table.py --images corpus/ --captions captions.json \\
        --terms terms.txt --preset natural --encoder dinov2_reg --out my_table.npz

``captions.json`` maps an image file name (relative to ``--images``) to its
caption, or to a list of captions. ``--terms`` is a comma-separated list or a
text file with one term per line. ``--forms`` optionally maps a term to the
surface forms that name it in captions (natural preset; by default a term is
named by itself and its plural). ``--preset`` sets every other constant to the
paper's value for that domain; individual flags override it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from captor import BuildConfig, Corpus, build_table, save_table
from captor.encoders import ENCODERS, load_encoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--terms", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset", default="natural", choices=("natural", "pathology"))
    ap.add_argument("--encoder", default=None, choices=ENCODERS,
                    help="default: dinov2_reg (natural) or conch (pathology)")
    ap.add_argument("--forms", type=Path, default=None)
    ap.add_argument("--strict-terms", default="",
                    help="comma-separated terms named only by their full name")
    ap.add_argument("--words", type=int, default=None, help="K")
    ap.add_argument("--vocabs", type=int, default=None, help="V")
    ap.add_argument("--em-rounds", type=int, default=None)
    ap.add_argument("--null-scale", dest="null_scale", action="store_true",
                    default=None, help="divide signatures by their permutation null")
    ap.add_argument("--no-null-scale", dest="null_scale", action="store_false")
    ap.add_argument("--no-joint-words", action="store_true")
    ap.add_argument("--tokens", type=Path, default=None,
                    help="where to keep the corpus tokens (.npy); reused if present")
    ap.add_argument("--corpus-name", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = BuildConfig.preset(a.preset)
    if a.words:
        cfg.words = a.words
    if a.vocabs:
        cfg.vocabs = a.vocabs
    if a.em_rounds is not None:
        cfg.em_rounds = a.em_rounds
    if a.null_scale is not None:
        cfg.null_scale = a.null_scale
    if a.no_joint_words:
        cfg.joint_words, cfg.trust_lambda = False, None
    cfg.strict_terms = [t.strip() for t in a.strict_terms.split(",") if t.strip()]
    encoder = a.encoder or ("dinov2_reg" if a.preset == "natural" else "conch")

    p = Path(a.terms)
    terms = (p.read_text().split("\n") if p.exists() else a.terms.split(","))
    terms = [t.strip() for t in terms if t.strip()]
    caps = json.loads(a.captions.read_text())
    names = [n for n in sorted(caps) if _readable(a.images / n)]
    if len(names) < len(caps):
        print(f"skipped {len(caps) - len(names)} missing or unreadable images")
    texts = [" ".join(c.lower() for c in ([caps[n]] if isinstance(caps[n], str)
                                          else caps[n])) for n in names]
    print(f"{len(names)} captioned images, {len(terms)} terms")

    # Natural corpora are kept in float16 and pathology corpora in float32, as
    # the pretrained tables were; pathology fields are brought to 224 px with
    # bicubic resampling for the corpus.
    natural = encoder == "dinov2_reg"
    side = a.tokens.with_suffix(".json") if a.tokens else None
    if a.tokens and a.tokens.exists():
        info = json.loads(side.read_text()) if side.exists() else {}
        if info.get("names") != names or info.get("encoder") != encoder:
            raise SystemExit(f"{a.tokens} was encoded from other images or another "
                             "encoder; remove it or pass another --tokens path")
        corpus = Corpus(np.load(a.tokens, mmap_mode="r"), texts)
    else:
        enc = load_encoder(encoder, device=a.device)
        corpus = Corpus.encode([a.images / n for n in names], texts, enc,
                               out=a.tokens, batch=a.batch,
                               dtype=np.float16 if natural else np.float32,
                               progress=lambda n: print(f"  encoded {n}", end="\r"),
                               **({} if natural else {"resample": Image.BICUBIC}))
        print()
        if side:
            side.write_text(json.dumps({"encoder": encoder, "names": names}))
    forms = json.loads(a.forms.read_text()) if a.forms else None
    table = build_table(corpus, terms, encoder, cfg, forms=forms, device=a.device,
                        corpus_name=a.corpus_name)
    out = a.out.with_suffix(".npz")
    save_table(table, out)
    print(f"wrote {out}: {len(table.terms)} terms, {table.n_vocabs} vocabularies "
          f"of {table.n_words} words")


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()

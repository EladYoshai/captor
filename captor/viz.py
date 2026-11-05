"""Drawing label maps and score maps over an image."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PIL import Image

# 20 distinguishable colours (Tableau 20), cycled for longer term lists
PALETTE = np.array([
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150),
    (197, 176, 213), (196, 156, 148), (247, 182, 210), (199, 199, 199),
    (219, 219, 141), (158, 218, 229)], dtype=np.uint8)


def crop_like_encoder(image: Image.Image, center_crop: bool) -> Image.Image:
    """The region the encoder sees: the centre square for natural images, the
    whole field for pathology."""
    if not center_crop:
        return image
    w, h = image.size
    s = min(w, h)
    return image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def colorize(labels: np.ndarray, size: Optional[tuple[int, int]] = None) -> Image.Image:
    """An RGB image of a label map, one colour per term index."""
    im = Image.fromarray(PALETTE[labels % len(PALETTE)])
    return im.resize(size, Image.NEAREST) if size else im


def overlay(image: Image.Image, labels: np.ndarray, alpha: float = 0.5,
            keep: Optional[np.ndarray] = None) -> Image.Image:
    """``image`` blended with its colour-coded label map.

    With a boolean ``keep`` mask on the label grid (for example
    ``Segmentation.confident()``), patches outside it are left uncoloured and faded.
    """
    image = image.convert("RGB")
    out = Image.blend(image, colorize(labels, image.size), alpha)
    if keep is None:
        return out
    faded = Image.blend(image, Image.new("RGB", image.size, (235, 235, 235)), 0.65)
    mask = Image.fromarray(np.asarray(keep, np.uint8) * 255).resize(image.size, Image.NEAREST)
    return Image.composite(out, faded, mask)


def heatmap(scores: np.ndarray, size: Optional[tuple[int, int]] = None) -> Image.Image:
    """A blue-to-red map of one term's scores, standardised over the image."""
    z = (scores - scores.mean()) / (scores.std() + 1e-12)
    t = np.clip((z + 2.5) / 5.0, 0.0, 1.0)
    rgb = np.stack([t, 1.0 - np.abs(2 * t - 1.0), 1.0 - t], -1)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return im.resize(size, Image.BICUBIC) if size else im


def legend(terms: Sequence[str]) -> list[tuple[str, tuple[int, int, int]]]:
    """(term, colour) pairs matching :func:`colorize`."""
    return [(t, tuple(int(c) for c in PALETTE[i % len(PALETTE)]))
            for i, t in enumerate(terms)]

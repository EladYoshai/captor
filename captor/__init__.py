"""CAPTOR: text-encoder-free grounding by counting caption co-occurrence over a
frozen image encoder's visual words."""
from .build import BuildConfig, Corpus, build_table
from .io import PRETRAINED, Table, load_table, resolve_table, save_table
from .model import CAPTOR, Segmentation, image_scores, patch_scores

__version__ = "1.0.0"
__all__ = ["CAPTOR", "Segmentation", "Table", "BuildConfig", "Corpus",
           "build_table", "load_table", "save_table", "resolve_table",
           "patch_scores", "image_scores", "PRETRAINED"]

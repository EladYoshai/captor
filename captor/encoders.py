"""Frozen image encoders and how their patch tokens are read.

=============  ==========================================  =====  =======  ======
name           weights                                     input  grid     dim
=============  ==========================================  =====  =======  ======
dinov2_reg     facebookresearch/dinov2, dinov2_vitb14_reg  196    14 x 14  768
conch          MahmoodLab/CONCH (gated)                    448    28 x 28  768
uni            MahmoodLab/UNI (gated)                      224    14 x 14  1024
phikon_v2      owkin/phikon-v2                             224    14 x 14  1024
=============  ==========================================  =====  =======  ======

Tokens are the encoder's patch tokens (prefix and register tokens removed),
each normalised to unit length. Pathology encoders read a field resized to
224 x 224 first; CONCH's own preprocessing then resizes it to 448.

CONCH and UNI are gated on the Hugging Face Hub: request access on their model
pages and log in (``huggingface-cli login``) before first use. CONCH also needs
its package: ``pip install git+https://github.com/Mahmoodlab/CONCH.git``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

ENCODERS = ("dinov2_reg", "conch", "uni", "phikon_v2")
PATHOLOGY_FIELD = 224


class Encoder:
    """Wraps a frozen model: PIL images in, unit-normalised tokens (B, P, D) out."""

    def __init__(self, name: str, forward: Callable, prepare: Callable,
                 device: str, field: Optional[int] = None,
                 resample: int = Image.BILINEAR):
        self.name, self.device = name, device
        self._forward, self._prepare = forward, prepare
        self.field, self.resample = field, resample

    def tokens(self, images: Sequence[Image.Image], batch: int = 32,
               resample: Optional[int] = None) -> np.ndarray:
        """Patch tokens of ``images``. ``resample`` overrides the PIL filter used
        to bring a pathology field to 224 x 224 (bilinear by default)."""
        import torch
        rs = self.resample if resample is None else resample
        out = []
        with torch.no_grad():
            for s in range(0, len(images), batch):
                ims = [im.convert("RGB") for im in images[s:s + batch]]
                if self.field:
                    ims = [im.resize((self.field, self.field), rs) for im in ims]
                f = self._forward(self._prepare(ims))
                out.append(torch.nn.functional.normalize(f.float(), dim=-1)
                           .cpu().numpy())
        return np.concatenate(out)


def load_encoder(name: str, device: str = "cpu", size: int = 196) -> Encoder:
    """Load one of :data:`ENCODERS` in evaluation mode on ``device``."""
    import torch
    if name == "dinov2_reg":
        import torchvision.transforms as T
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        m = m.eval().to(device)
        pre = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return Encoder(name,
                       lambda x: m.forward_features(x)["x_norm_patchtokens"],
                       lambda ims: torch.stack([pre(i) for i in ims]).to(device),
                       device)
    if name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        model, pre = create_model_from_pretrained("conch_ViT-B-16",
                                                  "hf_hub:MahmoodLab/conch")
        trunk = model.visual.trunk.eval().to(device)
        npre = getattr(trunk, "num_prefix_tokens", 1)
        return Encoder(name, lambda x: trunk.forward_features(x)[:, npre:],
                       lambda ims: torch.stack([pre(i) for i in ims]).to(device),
                       device, field=PATHOLOGY_FIELD)
    if name == "uni":
        import timm
        m = timm.create_model("hf_hub:MahmoodLab/uni", pretrained=True,
                              init_values=1e-5, dynamic_img_size=True)
        m = m.eval().to(device)
        cfg = timm.data.resolve_data_config(m.pretrained_cfg, model=m)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)

        def prep(ims):
            x = np.stack([np.asarray(im) for im in ims]).astype(np.float32) / 255.0
            return (torch.from_numpy(x).permute(0, 3, 1, 2).to(device) - mean) / std
        npre = getattr(m, "num_prefix_tokens", 1)
        return Encoder(name, lambda x: m.forward_features(x)[:, npre:], prep,
                       device, field=PATHOLOGY_FIELD)
    if name == "phikon_v2":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        m = AutoModel.from_pretrained("owkin/phikon-v2").eval().to(device)
        return Encoder(name, lambda x: m(pixel_values=x).last_hidden_state[:, 1:],
                       lambda ims: proc(images=list(ims), return_tensors="pt")
                       ["pixel_values"].to(device),
                       device, field=PATHOLOGY_FIELD)
    raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")

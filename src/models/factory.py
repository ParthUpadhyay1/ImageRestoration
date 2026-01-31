from typing import Dict, Any
import torch.nn as nn

from .nafnet import NAFNet


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    m = cfg["model"]
    return NAFNet(
        in_ch=m.get("in_ch", 3),
        out_ch=m.get("out_ch", 3),
        width=int(m.get("width", 32)),
        enc_blocks=tuple(m.get("enc_blocks", [2,2,4,4])),
        dec_blocks=tuple(m.get("dec_blocks", [2,2,2,2])),
        middle_blocks=int(m.get("middle_blocks", 4)),
        dropout=float(m.get("dropout", 0.0)),
    )

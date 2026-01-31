import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import load_yaml, ensure_dir
from .data.dataset import PairedRestorationDataset
from .models.factory import build_model


def _save_rgb01(path: str, img01: np.ndarray) -> None:
    img = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg["predict"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    data_root = cfg["paths"]["data_root"]
    exts = cfg["data"]["extensions"]
    crop = int(cfg["data"]["crop_size"])

    ds = PairedRestorationDataset(data_root, args.split, crop, exts, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

    model = build_model(cfg).to(device)
    model.eval()

    # load checkpoint
    ckpt_path = os.path.join(cfg["paths"]["artifacts_root"], "checkpoints", "best.pt" if cfg["predict"].get("use_best_checkpoint", True) else "last.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        print(f"WARNING: checkpoint not found at {ckpt_path}. Running with randomly initialized weights.")

    out_root = os.path.join(cfg["predict"]["save_dir"], args.split)
    ensure_dir(out_root)
    save_ext = cfg["predict"].get("save_format", "png").lower().strip(".")
    assert save_ext in ["png", "jpg", "jpeg"]

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict {args.split}"):
            x = batch["input"].to(device)
            name = batch["name"][0]
            pred = model(x)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # HWC RGB 0..1

            out_name = Path(name).stem + "." + save_ext
            _save_rgb01(os.path.join(out_root, out_name), pred)

    print(f"Saved predictions to: {out_root}")


if __name__ == "__main__":
    main()

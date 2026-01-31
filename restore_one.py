import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils import load_yaml
from src.models.factory import build_model


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_rgb(path: str, img_rgb: np.ndarray) -> None:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/default.yaml")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"], help="Which checkpoint to use")
    ap.add_argument("--input", required=True, help="Path to a single LQ image")
    ap.add_argument("--output", default="restored.png", help="Where to save restored image")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Choose device strictly based on availability + your config preference
    requested = cfg.get("predict", {}).get("device", "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Fix CUDA or set predict.device=cpu.")
    device = torch.device(requested if requested in ["cuda", "cpu"] else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build model
    model = build_model(cfg).to(device).eval()

    # Load checkpoint
    ckpt_path = Path(cfg["paths"]["artifacts_root"]) / "checkpoints" / f"{args.ckpt}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first to create it.")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    # Read + preprocess
    x = read_rgb(args.input).astype(np.float32) / 255.0
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # 1,C,H,W

    # Inference
    with torch.no_grad():
        pred = model(x_t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()  # H,W,C

    out = (pred * 255.0).round().astype(np.uint8)
    save_rgb(args.output, out)

    print(f"Saved restored image to: {args.output}")


if __name__ == "__main__":
    main()

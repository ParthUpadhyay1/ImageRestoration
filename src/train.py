import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import load_yaml, set_seed, ensure_dir, AverageMeter
from .data.dataset import PairedRestorationDataset
from .models.factory import build_model
from .metrics import psnr, ssim


def save_checkpoint(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int, best: bool) -> None:
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "best": best,
    }
    torch.save(ckpt, path)


def validate(model, loader, device):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            p = psnr(pred, y).mean().item()
            s = ssim(pred, y).mean().item()
            psnr_meter.update(p, n=x.size(0))
            ssim_meter.update(s, n=x.size(0))
    return psnr_meter.avg, ssim_meter.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 1337)))

    artifacts = cfg["paths"]["artifacts_root"]
    ensure_dir(artifacts)
    ensure_dir(os.path.join(artifacts, "checkpoints"))

    # snapshot config used
    Path(os.path.join(artifacts, "run_config.yaml")).write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    data_root = os.path.join(cfg["paths"]["data_root"])
    exts = cfg["data"]["extensions"]
    crop = int(cfg["data"]["crop_size"])

    train_ds = PairedRestorationDataset(data_root, "train", crop, exts, train=True)
    val_ds = PairedRestorationDataset(data_root, "val", crop, exts, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    l1_w = float(cfg["loss"].get("l1_weight", 1.0))
    ssim_w = float(cfg["loss"].get("ssim_weight", 0.0))

    best_psnr = -1.0
    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"].get("log_every", 50))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        loss_meter = AverageMeter()

        for batch in pbar:
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(x)
                loss = l1_w * F.l1_loss(pred, y)
                if ssim_w > 0:
                    loss = loss + ssim_w * (1.0 - ssim(pred, y).mean())

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            loss_meter.update(loss.item(), n=x.size(0))
            step += 1

            if step % log_every == 0:
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        v_psnr, v_ssim = validate(model, val_loader, device)
        is_best = v_psnr > best_psnr
        best_psnr = max(best_psnr, v_psnr)

        ckpt_dir = os.path.join(artifacts, "checkpoints")
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optim, epoch, best=False)
        if is_best:
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optim, epoch, best=True)

        print(f"[val] PSNR={v_psnr:.3f}  SSIM={v_ssim:.4f}  best_PSNR={best_psnr:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    pred, target: float tensors in [0,1], shape (N,C,H,W)
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=(1,2,3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    """
    Lightweight SSIM (not optimized), returns per-image SSIM.
    pred/target in [0,1], shape (N,C,H,W)
    """
    # gaussian window
    device = pred.device
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * (1.5**2)))
    g = (g / g.sum()).view(1, 1, -1)
    window = (g.transpose(1,2) @ g).unsqueeze(0)  # 1x1xWxW

    def filter2d(x):
        c = x.shape[1]
        w = window.expand(c, 1, window_size, window_size)
        return F.conv2d(x, w, padding=window_size//2, groups=c)

    mu_x = filter2d(pred)
    mu_y = filter2d(target)
    sigma_x = filter2d(pred * pred) - mu_x * mu_x
    sigma_y = filter2d(target * target) - mu_y * mu_y
    sigma_xy = filter2d(pred * target) - mu_x * mu_y

    c1 = (k1 ** 2)
    c2 = (k2 ** 2)

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.mean(dim=(1,2,3))

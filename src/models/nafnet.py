import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    A compact NAFNet-style block (good baseline for restoration).
    """
    def __init__(self, c: int, dropout: float = 0.0):
        super().__init__()
        dw = c * 2

        self.norm1 = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, dw, 1, 1, 0)
        self.dwconv = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(dw // 2, c, 1, 1, 0)

        self.norm2 = nn.BatchNorm2d(c)
        self.ffn1 = nn.Conv2d(c, dw, 1, 1, 0)
        self.ffn2 = nn.Conv2d(dw // 2, c, 1, 1, 0)

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scales
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.pw2(y)
        y = self.drop(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)
        y = self.ffn2(y)
        y = self.drop(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, width: int = 32,
                 enc_blocks=(2,2,4,4), dec_blocks=(2,2,2,2), middle_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_ch, 3, 1, 1)

        # encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for nb in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, dropout) for _ in range(nb)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan, dropout) for _ in range(middle_blocks)])

        # decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for nb in dec_blocks:
            self.ups.append(nn.ConvTranspose2d(chan, chan // 2, 2, 2))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, dropout) for _ in range(nb)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.intro(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # handle odd shapes
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
            x = dec(x)

        x = self.ending(x)
        # Residual learning: predict correction
        return torch.clamp(inp + x, 0.0, 1.0)

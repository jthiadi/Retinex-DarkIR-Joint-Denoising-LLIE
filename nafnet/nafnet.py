import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """
    Split channels in half, multiply — replaces ReLU/GELU
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, ffn_expand=2, dropout=0.0):
        super().__init__()

        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1,
            groups=channels
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        ffn_ch = int(channels * ffn_expand)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_ch * 2, 1),
            SimpleGate(),
            nn.Conv2d(ffn_ch, channels, 1)
        )

        self.norm1 = LayerNorm(channels)
        self.norm2 = LayerNorm(channels)

        self.beta = nn.Parameter(
            torch.ones(1, channels, 1, 1) * 1e-3
        )

        self.gamma = nn.Parameter(
            torch.ones(1, channels, 1, 1) * 1e-3
        )

        self.proj = nn.Conv2d(channels, channels, 1)

        self.gate = SimpleGate()

        self.proj2 = nn.Conv2d(
            channels // 1,
            channels,
            1
        )

    def forward(self, x):

        residual = x

        x = self.norm1(x)
        x = self.dw_conv(x)
        x = x * self.ca(x)
        x = self.proj(x)

        x = residual + x * self.beta

        residual = x

        x = self.norm2(x)
        x = self.ffn(x)

        return residual + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base_ch=32,
        enc_blocks=[2, 2, 4, 8],
        dec_blocks=[2, 2, 2, 2],
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_ch,
            base_ch,
            3,
            1,
            1
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = base_ch

        for num_blocks in enc_blocks:

            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(ch)
                        for _ in range(num_blocks)
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    ch,
                    ch * 2,
                    2,
                    2
                )
            )

            ch *= 2

        self.bottleneck = nn.Sequential(
            *[
                NAFBlock(ch)
                for _ in range(12)
            ]
        )

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num_blocks in dec_blocks:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 1),
                    nn.PixelShuffle(2)
                )
            )

            ch = ch // 2

            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(ch)
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.outro = nn.Conv2d(
            ch,
            out_ch,
            3,
            1,
            1
        )

    def forward(self, x):

        inp = x

        x = self.intro(x)

        skips = []

        for enc, down in zip(
            self.encoders,
            self.downs
        ):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(
            self.ups,
            self.decoders,
            reversed(skips)
        ):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.outro(x)

        return inp + x
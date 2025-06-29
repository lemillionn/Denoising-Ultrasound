import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ------------------------------------------------------
# 1D Residual Block with InstanceNorm and Dropout
# ------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(channels, affine=True)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

# ------------------------------------------------------
# 1D Self-Attention Block
# ------------------------------------------------------
class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels//8, 1)
        self.key   = nn.Conv1d(channels, channels//8, 1)
        self.value = nn.Conv1d(channels, channels,    1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, T = x.size()
        q = self.query(x).view(batch, -1, T)  # (B, C', T)
        k = self.key(x).view(batch, -1, T)
        attn = torch.softmax(torch.bmm(q.transpose(1,2), k), dim=-1)  # (B, T, T)
        v = self.value(x).view(batch, -1, T)
        out = torch.bmm(v, attn.transpose(1,2)).view(batch, C, T)
        return self.gamma * out + x

# ------------------------------------------------------
# 1D Residual U-Net with Self-Attention at Bottleneck
# ------------------------------------------------------
class ResUNet1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, depths=(1,2,4), dropout=0.1):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        ch = in_channels
        for d in depths:
            out_ch = base_channels * d
            self.downs.append(nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock1D(out_ch, dropout=dropout)
            ))
            ch = out_ch

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResBlock1D(ch, dropout=dropout),
            SelfAttention1D(ch),
            ResBlock1D(ch, dropout=dropout)
        )

        # Decoder
        self.ups = nn.ModuleList()
        for d in reversed(depths):
            out_ch = base_channels * d
            upsample = nn.Sequential(
                nn.ConvTranspose1d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
            project = nn.Sequential(
                nn.Conv1d(2*out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock1D(out_ch, dropout=dropout)
            )
            self.ups.append(nn.ModuleDict({"upsample": upsample, "project": project}))
            ch = out_ch

        # Final convolution
        self.final = nn.Conv1d(ch, in_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up in self.ups:
            skip = skips.pop()
            x = up['upsample'](x)
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                x = F.pad(x, (0, diff))
            x = up['project'](torch.cat([x, skip], dim=1))
        return self.final(x)

# ------------------------------------------------------
# Generator wrapper
# ------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, base_channels=32, depths=(1,2,4), dropout=0.1):
        super().__init__()
        self.net = ResUNet1D(in_channels=1,
                             base_channels=base_channels,
                             depths=depths,
                             dropout=dropout)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------
# Discriminator with Spectral Normalization
# ------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        layers = []
        ch = in_channels
        for mult in [1,2,4,8]:
            out_ch = base_channels * mult
            layers.append(spectral_norm(
                nn.Conv1d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out_ch
        # final patch output
        layers.append(spectral_norm(
            nn.Conv1d(ch, 1, kernel_size=4, stride=1, padding=1)
        ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

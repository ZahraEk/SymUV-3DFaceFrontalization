import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class GatedConv(nn.Module):
    """Gated convolution for masked or unreliable regions."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.gate = nn.Conv2d(in_ch, out_ch, k, s, p)
    def forward(self, x):
        return self.conv(x) * torch.sigmoid(self.gate(x))

class ResidualBottleneck(nn.Module):
    """Residual Bottleneck with dilated conv + InstanceNorm"""
    def __init__(self, ch, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, dilation, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(ch)
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + x)

class Generator(nn.Module):
    """U-Net + GatedConv + Residual Bottleneck"""
    def __init__(self, in_ch=7, base=64):
        super().__init__()
        self.e1 = GatedConv(in_ch, base)
        self.e2 = GatedConv(base, base*2, s=2)
        self.e3 = GatedConv(base*2, base*4, s=2)
        self.e4 = GatedConv(base*4, base*8, s=2)
        self.e5 = GatedConv(base*8, base*8, s=2)
        self.bottleneck = ResidualBottleneck(base*8, dilation=2)
        self.d5 = GatedConv(base*8 + base*8, base*8)
        self.d4 = GatedConv(base*8 + base*4, base*4)
        self.d3 = GatedConv(base*4 + base*2, base*2)
        self.d2 = GatedConv(base*2 + base, base)
        self.d1 = GatedConv(base + base, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        b = self.bottleneck(e5)
        d5 = self.d5(torch.cat([F.interpolate(b, e4.shape[2:], mode='bilinear', align_corners=False), e4],1))
        d4 = self.d4(torch.cat([F.interpolate(d5, e3.shape[2:], mode='bilinear', align_corners=False), e3],1))
        d3 = self.d3(torch.cat([F.interpolate(d4, e2.shape[2:], mode='bilinear', align_corners=False), e2],1))
        d2 = self.d2(torch.cat([F.interpolate(d3, e1.shape[2:], mode='bilinear', align_corners=False), e1],1))
        d1 = self.d1(torch.cat([d2, e1],1))
        return torch.tanh(self.out(d1))

class DiscriminatorBlock(nn.Module):
    """Single PatchGAN block with spectral normalization."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self,x): return self.block(x)

class Discriminator(nn.Module):
    """PatchGAN style Discriminator"""
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.model = nn.Sequential(
            DiscriminatorBlock(in_ch, base),
            DiscriminatorBlock(base, base*2),
            DiscriminatorBlock(base*2, base*4),
            DiscriminatorBlock(base*4, base*8),
            spectral_norm(nn.Conv2d(base*8,1,3,1,1))
        )
    def forward(self,x): return self.model(x)

class DualDiscriminator(nn.Module):
    """Global + Local Discriminator"""
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.global_d = Discriminator(in_ch, base)
        self.local_d = Discriminator(in_ch, base)
    def forward(self, x, patch=None):
        out_g = self.global_d(x)
        out_l = self.local_d(patch) if patch is not None else None
        return out_g, out_l

"""MSF-Net: HSI-derived pseudo-RGB and 5x5 regional spectral fusion."""
from __future__ import annotations
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class SEBlock2D(nn.Module):
    """Squeeze-and-excitation channel recalibration."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.fc(F.adaptive_avg_pool2d(x, 1).reshape(batch, channels))
        return x * weights.reshape(batch, channels, 1, 1)


class SEBasicBlock(nn.Module):
    """A torchvision ResNet34 BasicBlock followed by an SE module."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.se = SEBlock2D(block.conv2.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.block(x))


class SEResNet34(nn.Module):
    """ResNet34 with SE after each BasicBlock; stem/Stage 1 are frozen."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        stages = (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.stages = nn.ModuleList([
            nn.Sequential(*(SEBasicBlock(block) for block in stage)) for stage in stages
        ])
        self.freeze_early = bool(pretrained)
        if self.freeze_early:
            for parameter in self.stem.parameters():
                parameter.requires_grad = False
            # Freeze the pretrained ResNet weights in Stage 1, while retaining
            # trainable SE modules, consistent with the manuscript description.
            for unit in self.stages[0]:
                for parameter in unit.block.parameters():
                    parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_early:
            self.stem.eval()
            for unit in self.stages[0]:
                unit.block.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class SpectrumNet(nn.Module):
    """Shared 1D-CNN-BiGRU encoder for 25 regional mean/std spectra [B,25,2,256]."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(128, out_dim), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim != 4 or tuple(spectrum.shape[1:]) != (25, 2, 256):
            raise ValueError(f"Expected [B,25,2,256], got {tuple(spectrum.shape)}")
        batch, regions, channels, bands = spectrum.shape
        x = spectrum.reshape(batch * regions, channels, bands)
        x = self.conv(x).transpose(1, 2)
        x, _ = self.gru(x)
        region_features = self.projection(x.mean(dim=1))
        return region_features.reshape(batch, regions, -1).mean(dim=1)


class GatedFusion(nn.Module):
    """Adaptive element-wise gated fusion in a shared 128-D feature space."""
    def __init__(self):
        super().__init__()
        self.morphology_projection = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2)
        )
        self.spectral_projection = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.2)
        )
        self.gate = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.Sigmoid()
        )

    def forward(self, morphology: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        morphology = self.morphology_projection(morphology)
        spectral = self.spectral_projection(spectral)
        gate = self.gate(torch.cat((morphology, spectral), dim=1))
        return gate * morphology + (1.0 - gate) * spectral


class MSFNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.morphology = SEResNet34(pretrained=pretrained)
        self.morphology_projection = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.spectral = SpectrumNet(out_dim=128)
        self.fusion = GatedFusion()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, pseudo_rgb: torch.Tensor, regional_spectra: torch.Tensor) -> torch.Tensor:
        morphology = self.morphology_projection(self.morphology(pseudo_rgb))
        spectral = self.spectral(regional_spectra)
        return self.classifier(self.fusion(morphology, spectral))


def smoke_test(device: str = "cpu", batch_size: int = 2):
    model = MSFNet(pretrained=False).to(device)
    pseudo_rgb = torch.rand(batch_size, 3, 100, 100, device=device)
    regional = torch.rand(batch_size, 25, 2, 256, device=device)
    logits = model(pseudo_rgb, regional)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    F.cross_entropy(logits, labels).backward()
    print(f"PASS: logits={tuple(logits.shape)}, device={device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    smoke_test(args.device, args.batch_size)

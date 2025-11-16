# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNNFeature(nn.Module):
    """
    CNN feature extractor + classifier for MM-Fi data shaped (3,114,10)
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        # After two poolings: (3,114,10) -> (32,28,2)
        self._feat_dim = 32 * 28 * 2
        self.fc1 = nn.Linear(self._feat_dim, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        logits = self.fc_out(feat)
        if return_feat:
            return feat, logits
        return logits

    def get_feature_dim(self):
        return 128


class ResNet18Feature(nn.Module):
    """
    ResNet18 feature extractor + classifier for MM-Fi data shaped (3,114,10).
    Adapted from torchvision but simplified for 2D input.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (simplified for small input)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._feat_dim = 512

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # Downsample block
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Identity blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        logits = self.fc(feat)
        if return_feat:
            return feat, logits
        return logits

    def get_feature_dim(self):
        return self._feat_dim


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # Shortcut to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = F.relu(x)
        return x


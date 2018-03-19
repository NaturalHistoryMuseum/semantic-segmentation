import torch
from torch import nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownsampleBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=scale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(self.relu(x))
        x = self.conv2(x)
        x = self.bn2(self.relu(x))
        return self.max_pool1(x), x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.resize = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x, skip):
        x = self.resize(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.bn1(self.relu(x))
        x = self.conv2(x)
        return self.bn2(self.relu(x))


class DenseEmbedding(nn.Module):
    def __init__(self):
        super(DenseEmbedding, self).__init__()

        self.down1 = DownsampleBlock(3, 16)
        self.down2 = DownsampleBlock(16, 32)
        self.down3 = DownsampleBlock(32, 64)
        self.down4 = DownsampleBlock(64, 128)

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.up4 = UpsampleBlock(256, 64)
        self.up3 = UpsampleBlock(128, 32)
        self.up2 = UpsampleBlock(64, 16)
        self.up1 = UpsampleBlock(32, 8)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.conv1(x)
        x = self.bn1(self.relu(x))

        x = self.conv2(x)
        x = self.bn2(self.relu(x))

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return x


class SemanticInstanceSegmentation(nn.Module):
    def __init__(self):
        super(SemanticInstanceSegmentation, self).__init__()
        self.embedding = DenseEmbedding()
        self.conv_semantic = nn.Conv2d(8, 5, kernel_size=1)
        self.conv_instance = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        embedding = self.embedding(x)
        return self.conv_semantic(embedding), self.conv_instance(embedding)

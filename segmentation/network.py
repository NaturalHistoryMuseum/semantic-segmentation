import torch
from torch import nn
from torchvision.models import resnet18


class DenseEmbedding(nn.Module):
    def __init__(self):
        super(DenseEmbedding, self).__init__()

        pretrained = resnet18(pretrained=True)

        # split ResNet into individual layers to enable reconstruction losses
        self.downsample1 = nn.Sequential(pretrained.conv1, pretrained.bn1, pretrained.relu, pretrained.maxpool)
        self.downsample2 = nn.Sequential(pretrained.layer1, pretrained.layer2)

        self.atrous1 = nn.Conv2d(128, 128, kernel_size=1, groups=128)
        self.atrous2 = nn.Conv2d(128, 128, kernel_size=3, dilation=6, padding=6, groups=128)
        self.atrous3 = nn.Conv2d(128, 128, kernel_size=3, dilation=12, padding=12, groups=128)
        self.atrous4 = nn.Conv2d(128, 128, kernel_size=3, dilation=18, padding=18, groups=128)

        self.conv1 = nn.Conv2d(640, 128, kernel_size=1)

    @staticmethod
    def global_avg_pool2d(x):
        return x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

    @staticmethod
    def global_features(x):
        return DenseEmbedding.global_avg_pool2d(x).expand_as(x)

    def forward(self, x, corrupted=False, variance=1):
        z1 = self.downsample1(x)
        if corrupted:
            z1 += variance * torch.randn_like(z1)
        z2 = self.downsample2(z1)

        convs = (self.atrous1, self.atrous2, self.atrous3, self.atrous4, self.global_features)
        atrous_pyramid = torch.cat([conv(z2) for conv in convs], dim=1)

        return z1, z2, self.conv1(atrous_pyramid)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.resize = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x, x_prev):
        x_prev = self.resize(x_prev)
        #print(x.size(),x_prev.size())
        x = torch.cat([x, x_prev], dim=1)
        x = self.conv(x)
        return self.bn(self.relu(x))


class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()

        self.upsample1 = UpsampleBlock(64 + 3, 3, scale_factor=4)
        self.upsample2 = UpsampleBlock(128 + 64, 64, scale_factor=2)

    def forward(self, x_tilde, z_tilde1, z_hat2):
        z_hat1 = self.upsample2(z_tilde1, z_hat2)
        x_hat = self.upsample1(x_tilde, z_hat1)
        return z_hat1, x_hat


class SemanticInstanceSegmentation(nn.Module):
    def __init__(self, variance=0.1):
        # number 4 on nn.Conv2d corresponds to classes
        #**************************************************
        # convert to parameter directory structure
        #**************************************************
        classes=5
        super(SemanticInstanceSegmentation, self).__init__()
        self.variance = variance
        self.embedding = DenseEmbedding()
        self.reconstruction = Reconstruction()
        self.conv_semantic = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
                                           nn.ReLU(),
                                           nn.BatchNorm2d(128),
                                           nn.Conv2d(128, classes, kernel_size=1),
                                           nn.Upsample(scale_factor=8, mode='bilinear'))
        self.conv_instance = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
                                           nn.ReLU(),
                                           nn.BatchNorm2d(128),
                                           nn.Conv2d(128, 2, kernel_size=1),
                                           nn.Upsample(scale_factor=8, mode='bilinear'))

    def forward_clean(self, x):
        z1, _, embedding = self.embedding(x)
        return z1, self.conv_semantic(embedding), self.conv_instance(embedding)

    def forward(self, x):
        x_tilde = x + self.variance * torch.randn_like(x)
        z_tilde1, z_hat2, embedding = self.embedding(x_tilde, corrupted=True, variance=self.variance)
        #print (x_tilde.dim(), z_tilde1.dim(), z_hat2.dim())
        #print (x_tilde.size(), z_tilde1.size(), z_hat2.size())
        z_hat1, x_hat = self.reconstruction(x_tilde, z_tilde1, z_hat2)
        return z_hat1, x_hat, self.conv_semantic(embedding), self.conv_instance(embedding)

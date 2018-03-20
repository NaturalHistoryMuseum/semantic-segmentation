import torch
from torch import nn
from torchvision.models import resnet18


class DenseEmbedding(nn.Module):
    def __init__(self):
        super(DenseEmbedding, self).__init__()

        self.pretrained = nn.Sequential(*list(resnet18(pretrained=True).children())[:-4])
        self.atrous1 = nn.Conv2d(128, 128, kernel_size=1, groups=128)
        self.atrous2 = nn.Conv2d(128, 128, kernel_size=3, dilation=6, padding=6, groups=128)
        self.atrous3 = nn.Conv2d(128, 128, kernel_size=3, dilation=12, padding=12, groups=128)
        self.atrous4 = nn.Conv2d(128, 128, kernel_size=3, dilation=18, padding=18, groups=128)
        self.pool = nn.AvgPool2d((32, 96))

        self.conv1 = nn.Conv2d(640, 128, kernel_size=1)

    def forward(self, x):
        x = self.pretrained(x)

        convs = (self.atrous1, self.atrous2, self.atrous3, self.atrous4, lambda x: self.pool(x).expand(-1, -1, 32, 96))
        atrous_pyramid = torch.cat([conv(x) for conv in convs], dim=1)

        return self.conv1(atrous_pyramid)


class SemanticInstanceSegmentation(nn.Module):
    def __init__(self):
        super(SemanticInstanceSegmentation, self).__init__()
        self.embedding = DenseEmbedding()
        self.conv_semantic = nn.Sequential(nn.Conv2d(128, 5, kernel_size=1),
                                           nn.Upsample(scale_factor=8, mode='bilinear'))
        self.conv_instance = nn.Sequential(nn.Conv2d(128, 2, kernel_size=1),
                                           nn.Upsample(scale_factor=8, mode='bilinear'))

    def forward(self, x):
        embedding = self.embedding(x)
        return self.conv_semantic(embedding), self.conv_instance(embedding)

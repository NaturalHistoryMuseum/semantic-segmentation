from itertools import islice
import logging
from pathlib import Path

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms

from datasets import Slides


logging.basicConfig(format='[%(asctime)s] %(message)s', filename='training.log', filemode='w', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


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


class SemanticSegmentation(nn.Module):
    def __init__(self):
        super(SemanticSegmentation, self).__init__()

        self.down1 = DownsampleBlock(3, 16)
        self.down2 = DownsampleBlock(16, 32)
        self.down3 = DownsampleBlock(32, 64)
        self.down4 = DownsampleBlock(64, 128)

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.up4 = UpsampleBlock(256, 64)
        self.up3 = UpsampleBlock(128, 32)
        self.up2 = UpsampleBlock(64, 16)
        self.up1 = UpsampleBlock(32, 8)

        self.conv3 = nn.Conv2d(8, 5, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.conv1(x)
        x = self.bn1(self.relu(x))

        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(self.relu(x))

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return self.conv3(x)


def visualise_segmentation(output, image, predicted_class, colours, n=5, dpi=250):
    gs = gridspec.GridSpec(2, n, width_ratios=[1]*n, wspace=0.1, hspace=0, top=0.95, left=0.17, right=0.845)
    fig = plt.figure(figsize=(n, 2))

    for i in range(n):
        plt.subplot(gs[0, i])
        plt.imshow(image.data[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.subplot(gs[1, i])
        class_image = np.zeros((predicted_class.shape[2], predicted_class.shape[3], 3))
        prediction = predicted_class[i, 0].cpu().numpy()
        for j in range(len(colours)):
            class_image[prediction == j] = colours[j]
        plt.imshow(class_image / 255)
        plt.axis('off')
    plt.savefig(str(output), dpi=dpi, bbox_inches='tight')


def train(model, train_loader, test_loader):
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 800, gamma=0.1)

    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}

    for epoch in range(3200):
        scheduler.step()

        if epoch % scheduler.step_size == 0:
            logging.debug(f'Learning rate set to {scheduler.get_lr()[0]}')

        model.train()

        for i, (image, target) in enumerate(train_loader):
            image, target = Variable(image).cuda(), Variable(target).cuda()
            optimizer.zero_grad()

            logits = model(image)
            logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
            loss = cross_entropy(logits_per_pixel.view(-1, 5), target.view(-1))

            loss.backward()
            optimizer.step()

            predicted_class = logits.data.max(1, keepdim=True)[1]
            accuracy = predicted_class.eq(target.data.view_as(predicted_class)).cpu().sum() / np.prod(predicted_class.shape)

            losses['train'].append(loss.data[0])
            accuracies['train'].append(accuracy)
            logging.debug(f'Epoch: {epoch:{3}}, Batch: {i:{3}}, Cross-entropy loss: {loss.data[0]}, Accuracy: {(accuracy * 100)}%')

        model.eval()

        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for image, target in islice(test_loader, 2):
                image, target = Variable(image).cuda(), Variable(target).cuda()

                logits = model(image)
                logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
                loss = cross_entropy(logits_per_pixel.view(-1, 5), target.view(-1))
                total_loss += loss.data[0]

                predicted_class = logits.data.max(1, keepdim=True)[1]
                accuracy = predicted_class.eq(target.data.view_as(predicted_class)).cpu().sum() / np.prod(predicted_class.shape)
                total_accuracy += accuracy

        average_loss = total_loss / 2
        average_accuracy = total_accuracy / 2
        losses['test'].append(average_loss)
        accuracies['test'].append(average_accuracy)
        logging.info(f'Epoch: {epoch:{3}}, Test Set, Cross-entropy loss: {average_loss}, Accuracy: {(average_accuracy * 100)}%')

        if (epoch + 1) % 5 == 0:
            visualise_segmentation(Path('results') / f'epoch_{epoch + 1}.png', image, predicted_class, colours=data.colours)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), Path('models') / f'epoch_{epoch + 1}')


if __name__ == '__main__':
    model = SemanticSegmentation().cuda()

    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()])

    target_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long())])

    data = Slides(download=True, train=True, root='data', transform=transform, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=5, drop_last=True, shuffle=True)  # don't use multiple workers! Doesn't work with setting random seed
    data = Slides(download=True, train=False, root='data', transform=transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=5, drop_last=True, shuffle=True)  # don't use multiple workers! Doesn't work with setting random seed

    # train(model, train_loader, test_loader)
    model.load_state_dict(torch.load('models/epoch_3200'))

    model.eval()
    transform = transforms.ToTensor()
    images = torch.stack([F.pad(transform(Image.open(filename)), (112, 112, 91, 91)) for filename in sorted((Path('data') / 'raw' / 'images').glob('*.JPG'))[:12]], dim=0)
    with torch.no_grad():
        for i in range(5):
            image = images[(i * 2):((i + 1) * 2)].cuda()
            logits = model(image)
            predicted_class = F.pad(logits.data.max(1, keepdim=True)[1], (-112, -112, -91, -91))
            image = F.pad(image, (-112, -112, -91, -91))
            visualise_segmentation(Path() / f'full_{i}.png', image, predicted_class, colours=data.colours, n=2, dpi=750)

    images = torch.stack([F.pad(transform(Image.open(filename)), (112, 112, 91, 91)) for filename in sorted(Path('test').glob('*.JPG'))], dim=0)
    with torch.no_grad():
        for i in range(1):
            image = images[(i * 7):((i + 1) * 7)].cuda()
            logits = model(image)
            predicted_class = F.pad(logits.data.max(1, keepdim=True)[1], (-112, -112, -91, -91))
            image = F.pad(image, (-112, -112, -91, -91))
            visualise_segmentation(Path() / f'test_{i}.png', image, predicted_class, colours=data.colours, n=7, dpi=750)

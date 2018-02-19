from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from datasets import Slides
from network import SemanticSegmentation


def mean(iterable):
    iterator = iter(iterable)
    total = next(iterator)
    terms = 1
    for term in iterator:
        total += term
        terms += 1
    return total / terms


def hinge(distance, min_distance=None, max_distance=None):
    if min_distance is not None and max_distance is not None:
        raise ValueError('can only hinge from above or below, not both')

    if min_distance is not None:
        return F.relu(distance - min_distance)

    if max_distance is not None:
        return F.relu(max_distance - distance)


class InstanceSegmentation(nn.Module):
    def __init__(self):
        super(InstanceSegmentation, self).__init__()
        self.pretrained = SemanticSegmentation()
        self.conv = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        return self.conv(self.pretrained.embedding(x))


class Clustering:
    def __init__(self, embedding, cluster):
        self.embedding = embedding
        self.cluster = cluster
        self.C = cluster.max() + 1

    def __getitem__(self, index):
        return self.embedding[:, :, self.cluster == index]

    def __iter__(self):
        for index in range(self.C):
            yield self[index]


class DiscriminativeLoss(nn.Module):
    def __init__(self, alpha=1, beta=2, gamma=0.001, delta_v=0.5, delta_distance=1.5):
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_distance = delta_distance
        self.delta_v = delta_v

    def variance_loss(self, x, embeddings):
        return mean((hinge(((x - embedding.mean(dim=2, keepdim=True).expand_as(x))**2).sum(dim=1),
                           min_distance=self.delta_v)**2).mean()
                    for embedding in embeddings)

    def distance_loss(self, embeddings):
        return mean(hinge(((embedding_A.mean(dim=2) - embedding_B.mean(dim=2))**2).sum(),
                          max_distance=(2 * self.delta_distance))**2
                    for embedding_A, embedding_B in combinations(embeddings, 2))

    def regularization_loss(self, embeddings):
        return mean((embedding.mean(dim=2)**2).sum() for embedding in embeddings)

    def forward(self, x, embeddings):
        return (self.alpha * self.variance_loss(x, embeddings) +
                self.beta * self.distance_loss(embeddings) +
                self.gamma * self.regularization_loss(embeddings))


def instances_from_colors(image):
    image_colors = image.reshape(-1, 3)
    colors, counts = np.unique(image_colors, axis=0, return_counts=True)
    unique_colors = colors[(counts / (image.size // 3)) > 0.01]
    distances = np.zeros((image_colors.shape[0], unique_colors.shape[0]))
    for i, color in enumerate(unique_colors):
        distances[:, i] = ((image_colors - color[np.newaxis])**2).sum(axis=1)
    return np.argmin(distances, axis=1).reshape(image.shape[:2])


def mean_shift(label_embedding):
    neigh = NearestNeighbors(radius=0.5, metric='euclidean')
    neigh.fit(label_embedding.copy().T)

    predicted_instances = -np.ones(label_embedding.shape[1])
    unlabeled = np.where(predicted_instances < 0)[0]

    while unlabeled.size > 0:
        index = np.random.choice(unlabeled)
        centre = label_embedding.T[index]
        for i in range(100):
            neighbors = neigh.radius_neighbors(centre.reshape(1, 2), return_distance=False)[0]
            new_centre = label_embedding.T[neighbors].mean(axis=0)
            if np.allclose(centre, new_centre):
                break
            centre = new_centre
        neighbours = neigh.radius_neighbors(centre.reshape(1, 2), return_distance=False)[0]

        centre_index = neigh.kneighbors(centre.reshape(1, 2), n_neighbors=1, return_distance=False)[0][0]
        predicted_instances[neighbours] = centre_index
        predicted_instances[index] = centre_index
        unlabeled = np.where(predicted_instances < 0)[0]
    return predicted_instances


data = Slides(download=True, train=True, root='data')

model = InstanceSegmentation().cuda()
clustering = DiscriminativeLoss().cuda()

imgs = [torch.Tensor(np.asarray(Image.open(filename)).transpose(2, 0, 1)[np.newaxis])
        for filename in ['image.png', 'image2.png']]
img_instances = [torch.Tensor(instances_from_colors(np.asarray(Image.open(filename)).astype(np.uint8))).long()
                 for filename in ['image_instances.png', 'image_instances2.png']]

train = False

if train:
    model.pretrained.load_state_dict(torch.load(Path('models') / 'epoch_3200'))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.1)
    model.train()
    losses = np.zeros(450)

    for iteration in range(450):
        scheduler.step()
        optimizer.zero_grad()

        for img, instances in zip(imgs, img_instances):
            img = img.cuda()
            instances = instances.cuda()

            embedding = model(Variable(img)).view(1, 2, -1)
            mask = instances.view(-1) > 0
            label_embedding = embedding[..., mask]
            label_instances = instances.view(-1)[mask] - 1

            loss = clustering(label_embedding, Clustering(label_embedding, label_instances))

            print(iteration, loss.data[0])
            losses[iteration] = loss.data[0]
            loss.backward()

        optimizer.step()

    torch.save(model.state_dict(), Path('models') / 'instance_model')
else:
    model.load_state_dict(torch.load(Path('models') / 'instance_model'))

    img = imgs[0].cuda()
    instances = img_instances[0].cuda()

    embedding = model(Variable(img)).view(1, 2, -1)
    mask = instances.view(-1) > 0
    label_embedding = embedding[..., mask]
    label_instances = instances.view(-1)[mask] - 1

label_embedding = label_embedding.data.cpu().numpy()[0]
label_instances = label_instances.cpu().numpy()

predicted_instances = mean_shift(label_embedding)

ax = plt.gca()
for index in np.unique(predicted_instances):
    cluster = label_embedding[:, predicted_instances == index]
    plt.plot(cluster[0], cluster[1], '+', label=index)
    ax.add_patch(plt.Circle((cluster[0].mean(), cluster[1].mean()), radius=1.5, fill=False, linestyle='--'))
    ax.add_patch(plt.Circle((cluster[0].mean(), cluster[1].mean()), radius=0.5, fill=False, linestyle='--'))

plt.legend()
plt.show()

predicted_image = np.zeros(512 * 512)
_, predicted_indices = np.unique(predicted_instances, return_inverse=True)
predicted_image[instances.view(-1).cpu().numpy() > 0] = predicted_indices + 1
plt.imshow(predicted_image.reshape(512, 512))
plt.show()
